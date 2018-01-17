#include <ngstd.hpp>
using namespace ngstd;
#include <bla.hpp>
using namespace ngbla;
#include <comp.hpp>
using namespace ngcomp;

// #include "complex_mat.hpp"

#include "h1.hpp"

#include "h1amg.hpp"

#include "concurrentqueue.h"
typedef moodycamel::ConcurrentQueue<size_t> TQueue;
typedef moodycamel::ProducerToken TPToken;
typedef moodycamel::ConsumerToken TCToken;


namespace h1amg
{
  static TQueue queue;


  H1AMG::H1AMG(const PDE& a_pde, const Flags& a_flags, const string a_name)
    : H1AMG(a_pde.GetBilinearForm(a_flags.GetStringFlag("bilinearform", "")), a_flags, a_name)
  { }

  H1AMG::H1AMG(shared_ptr<BilinearForm> a_bfa, const Flags& a_flags, const string a_name)
    : Preconditioner(a_bfa, a_flags, a_name), bfa(a_bfa), m_ndof(bfa->GetFESpace()->GetNDof()),
      dof_pair_weights(2*m_ndof), m_hashlocks(2*m_ndof)
  {
    if (bfa) {
      while (bfa->GetLowOrderBilinearForm()) {
        bfa = bfa->GetLowOrderBilinearForm();
      }
    }
    m_h1_options.maxlevel = int(flags.GetNumFlag("levels", 10));
    m_h1_options.level = int(flags.GetNumFlag("levels", 10));
    m_h1_options.variable_vcycle = flags.GetDefineFlag("variable_vcycle");
    m_h1_options.smoothed = flags.GetDefineFlag("smoothed");
    m_h1_options.semi_smoothed = !flags.GetDefineFlag("not_semi_smoothed");
    m_h1_options.special_level = int(flags.GetNumFlag("special_level", 3));
  }

  void H1AMG::InitLevel(shared_ptr<BitArray> afreedofs)
  {
    static Timer Tinit_level("H1-AMG::InitLevel");
    RegionTimer Rinit_level(Tinit_level);
    *testout << "initlevel amg" << endl;
    freedofs = afreedofs;
    int nr_dofs = freedofs->Size();
    weights_vertices = Array<double>(nr_dofs);
    weights_vertices = 0;
  }

  void H1AMG::FinalizeLevel(const BaseMatrix* mat)
  {
    static Timer Tfinlevel("H1-AMG::FinalizeLevel");
    RegionTimer Rfinlevel(Tfinlevel);

    *testout << "finalize lvl  amg" << endl;

    static Timer Tcreate_e2v("H1-AMG::FinalizeLevel::CreateE2VMapping");
    Tcreate_e2v.Start();

    size_t cnt = par_dof_pair_weights.Used();
    Array<double> weights (cnt);
    Array<INT<2> > edge_to_vertices (cnt);

    par_dof_pair_weights.IterateParallel
      ([&weights,&edge_to_vertices] (size_t i, INT<2> key, double weight)
       {
         weights[i] = weight;
         edge_to_vertices[i] = key;
       });
  
    Tcreate_e2v.Stop();

    amg_matrix = BuildAMGMat(dynamic_pointer_cast<SparseMatrixTM<double>>(const_cast<BaseMatrix*>(mat)->shared_from_this()),
                             edge_to_vertices,
                             weights, weights_vertices,freedofs, m_h1_options);

    cout << IM(5) << "matrices done" << endl;

    if (timing) { Timing(); }
    if (test) { Test(); }
  }

  void H1AMG::AddElementMatrixCommon(
                                     FlatArray<int> dnums, const FlatMatrix<double>& elmat, LocalHeap& lh)
  {
    static Timer addelmat("amg - addelmat",2);
    static Timer addelmat1("amg - addelmat1",2);
    static Timer addelmat2("amg - addelmat2",2);
    static Timer addelmat3("amg - addelmat3",2);

    int tid = TaskManager::GetThreadId();
    ThreadRegionTimer reg(addelmat,tid);
    NgProfiler::StartThreadTimer (addelmat1, tid);
  
    auto ndof = elmat.Height();
    FlatMatrix<double> constant_elmat(ndof, lh);
    FlatMatrix<double> nullspace_elmat(ndof, lh);

    auto vertex_weight = SplitMatrixConstant(elmat, constant_elmat, nullspace_elmat);

    if (vertex_weight != 0.0) {
      vertex_weight /= ndof;
      for (auto i=0; i < ndof; ++i)
        weights_vertices[ dnums[i] ] += vertex_weight;
    }

    // FlatMatrix<double> approx_elmat(ndof, lh);
    // approx_elmat = 0;

    NgProfiler::StopThreadTimer (addelmat1, tid);

    FlatMatrix<double> schur_complement(2, lh);
    BitArray used(ndof, lh);
    for (auto i=0; i < ndof; ++i) {
      for (auto j=0; j < ndof; ++j) {
        auto first_dof = dnums[i];
        auto second_dof = dnums[j];
        if (first_dof < second_dof) {
          // schur_complement = 0;
          used.Clear();

          // the schur-complement is calculated in respect to the two current
          // dofs
          used.Set(i);
          used.Set(j);

          {
            ThreadRegionTimer reg(addelmat2,tid);
            CalcSchurComplement(nullspace_elmat, schur_complement, used, lh);
          }
          double schur_entry = schur_complement(0);

          if (!std::isnan(schur_entry)) {
            INT<2> i2(first_dof, second_dof);
            // ThreadRegionTimer reg(addelmat3,tid);
            par_dof_pair_weights.Do (i2, [schur_entry] (auto & v) { v += schur_entry; });
          }
          else {
            INT<2> i2(first_dof, second_dof);
            par_dof_pair_weights.Do (i2, [schur_entry] (auto & v) { v += 0.0; });          
          }
        }
      }
    }

  }

  void H1AMG::AddElementMatrix(
                               FlatArray<int> dnums, const FlatMatrix<double>& elmat, ElementId ei, LocalHeap& lh)
  {
    HeapReset hr(lh);
    AddElementMatrixCommon(dnums, elmat, lh);
  }


  void H1AMG::AddElementMatrix(
                               FlatArray<int> dnums, const FlatMatrix<Complex>& elmat, ElementId ei, LocalHeap& lh)
  {
    HeapReset hr(lh);
    // auto combined_elmat = AddedParts(elmat, lh);
    FlatMatrix<double> combined_elmat(elmat.Height(), elmat.Width(), lh);
    combined_elmat = Real(elmat)+Imag(elmat);
    AddElementMatrixCommon(dnums, combined_elmat, lh);
  }

  shared_ptr<H1AMG_Mat> H1AMG::BuildAMGMat(shared_ptr<SparseMatrixTM<double>> sysmat,
                                           const Array<INT<2>>& edge_to_vertices,
                                           const Array<double>& weights_edges,
                                           const Array<double>& weights_vertices,
                                           shared_ptr<BitArray> free_dofs, const H1Options& h1_options)
  {
    static Timer Tbuild_h1("H1-AMG::BuildH1AMG");
    RegionTimer Rbuild_h1(Tbuild_h1);

    cout << IM(5) << "H1 Sysmat nze: " << sysmat->NZE() << endl;
    cout << IM(5) << "H1 Sysmat nze per row: " << sysmat->NZE() / (double)sysmat->Height() << endl;
    auto ne = edge_to_vertices.Size();
    auto nv = weights_vertices.Size();

    Array<int> vertex_coarse;

    Array<double> vertex_strength;
    Array<double> edge_collapse_weight;
    Array<double> vertex_collapse_weight;

    Array<INT<2> > coarse_edge_to_vertices;
    Array<int> edge_coarse;

    Array<double> weights_edges_coarse;
    Array<double> weights_vertices_coarse;

    ComputeCollapseWeights(
                           edge_to_vertices, weights_edges, weights_vertices, vertex_strength, edge_collapse_weight,
                           vertex_collapse_weight);

    // IterativeCollapse(
    //     edge_to_vertices, edge_collapse_weight, vertex_collapse_weight, free_dofs, edge_collapse,
    //     vertex_collapse);

    /*
      static Timer Tdist1sorted("Dist1 Sorted Collapsing");
      Tdist1sorted.Start();
      static Timer t1("Dist1 Sorted Collapsing sorting");
      static Timer t2("Dist1 Sorted Collapsing work");

      Dist1Collapser collapser(nv, ne);

      t1.Start();
      Array<int> indices(ne);
      Array<Edge> edges(ne);
      ParallelFor (ne, [&] (size_t edge)
      {
      indices[edge] = edge;
      edges[edge] = Edge(edge, edge_to_vertices[edge][0], edge_to_vertices[edge][1]);
      });

      ngstd::SampleSortI(edge_collapse_weight, indices);
      t1.Stop();

      t2.Start();
      int vcnt = 0;
      for (int i = ne-1; i >= 0; --i) {
      auto edge = edges[indices[i]];

      if (vcnt >= nv/2.)
      break;
      if (edge_collapse_weight[edge.id] >= 0.01 && !collapser.AnyVertexCollapsed(edge)) {
      ++vcnt;
      collapser.CollapseEdge(edge);
      }
      }
      t2.Stop();
      Tdist1sorted.Stop();
    */


    Array<int> indices(ne);
    ParallelFor (ne, [&] (size_t edge)
                 {
                   indices[edge] = edge;
                 });

    ngstd::SampleSortI(edge_collapse_weight, indices);
    Array<int> invindices(ne);
    ParallelFor (ne, [&] (size_t edge)
                 {
                   invindices[indices[edge]] = edge;
                 });
  
  
    static Timer Tdist1sorted("Dist1 Sorted Collapsing");
    Tdist1sorted.Start();

    Array<bool> vertex_collapse(nv);
    Array<bool> edge_collapse(ne);
    edge_collapse = false;
    vertex_collapse = false;

    TableCreator<int> v2e_creator(nv);
    for ( ; !v2e_creator.Done(); v2e_creator++)
      ParallelFor (ne, [&] (size_t e)
                   {
                     for (int j = 0; j < 2; j++)
                       v2e_creator.Add (edge_to_vertices[e][j], e);
                   });
    Table<int> v2e = v2e_creator.MoveTable();

    ParallelFor (v2e.Size(), [&] (size_t vnr)
                 {
                   // QuickSortI (edge_collapse_weight, v2e[vnr]);
                   QuickSortI (invindices, v2e[vnr]);
                 }, TasksPerThread(5));
  
    // build edge dependency
    TableCreator<int> edge_dag_creator(ne);
    for ( ; !edge_dag_creator.Done(); edge_dag_creator++)  
      ParallelFor (v2e.Size(), [&] (size_t vnr)
                   {
                     auto vedges = v2e[vnr];
                     for (int j = 0; j+1 < vedges.Size(); j++)
                       edge_dag_creator.Add (vedges[j+1], vedges[j]);
                   }, TasksPerThread(5));
    Table<int> edge_dag = edge_dag_creator.MoveTable();

    RunParallelDependency (edge_dag,
                           [&] (int edgenr)
                           {
                             auto v0 = edge_to_vertices[edgenr][0];
                             auto v1 = edge_to_vertices[edgenr][1];
                             if (edge_collapse_weight[edgenr] >= 0.01 && !vertex_collapse[v0] && !vertex_collapse[v1])
                               {
                                 edge_collapse[edgenr] = true;
                                 vertex_collapse[v0] = true;
                                 vertex_collapse[v1] = true;
                               }
                           });
    Tdist1sorted.Stop();

    vertex_collapse = false;
    for (int e = 0; e < ne; e++)
      if (edge_collapse[e])
        {
          auto v0 = edge_to_vertices[e][0];
          auto v1 = edge_to_vertices[e][1];
          vertex_collapse[max2(v0,v1)] = true;
        }
  
    /*
     *testout << "edge_weights = " << edge_collapse_weight << endl;  
     *testout << "edge_collapse = " << edge_collapse << endl;
     */
  
    int nr_coarse_vertices = ComputeFineToCoarseVertex(
                                                       edge_to_vertices, nv, edge_collapse, vertex_collapse, vertex_coarse);

    ComputeFineToCoarseEdge(edge_to_vertices, vertex_coarse, edge_coarse, coarse_edge_to_vertices);

    ComputeCoarseWeightsEdges(
                              edge_to_vertices, coarse_edge_to_vertices, edge_coarse, weights_edges, weights_edges_coarse);

    ComputeCoarseWeightsVertices(edge_to_vertices, vertex_coarse, nr_coarse_vertices, weights_edges,
                                 weights_vertices, weights_vertices_coarse);

    static Timer Tblock_table("H1-AMG::BlockJacobiTable");
    Tblock_table.Start();

    Array<int> nentries(nr_coarse_vertices);
    nentries = 0;

    for (auto fvert = 0; fvert < nv; ++ fvert) {
      auto cvert = vertex_coarse[fvert];
      if (cvert != -1) {
        nentries[cvert] += 1;
      }
    }

    auto blocks = make_shared<Table<int>>(nentries);
    Array<int> cnt(nr_coarse_vertices);
    cnt = 0;

    for (auto fvert = 0; fvert < nv; ++ fvert) {
      auto cvert = vertex_coarse[fvert];
      if (cvert != -1) {
        (*blocks)[cvert][cnt[cvert]++] = fvert;
      }
    }
    auto bjacobi = sysmat->CreateBlockJacobiPrecond(blocks, 0, 1, free_dofs);
    Tblock_table.Stop();

    SPtrSMdbl prol;
    int level_diff = h1_options.maxlevel - h1_options.level;
    if (h1_options.smoothed && level_diff % h1_options.special_level == 0) {
      prol = H1SmoothedProl(
                            vertex_coarse, nr_coarse_vertices, edge_to_vertices, weights_edges, sysmat->IsComplex());
    } else if (h1_options.semi_smoothed && level_diff % h1_options.special_level == 0) {
      auto triv_prol = CreateProlongation(vertex_coarse, nr_coarse_vertices, sysmat->IsComplex());
      prol = CreateSmoothedProlongation(edge_to_vertices, weights_edges, nv, move(triv_prol));
    }
    else {
      prol = CreateProlongation(vertex_coarse, nr_coarse_vertices, sysmat->IsComplex());
    }

    // build coarse mat
    static Timer Trestrict_sysmat("H1-AMG::RestrictSysmat");
    Trestrict_sysmat.Start();
    auto coarsemat = dynamic_pointer_cast<SparseMatrixTM<double>>(sysmat->Restrict(*prol));
    Trestrict_sysmat.Stop();

    cout << IM(5) << "H1 level " << h1_options.level
         << ", Nr. vertices: " << nv << ", Nr. edges: " << ne << endl
         << "e/v: " << ne/double(nv) << endl
         << "coarse/fine verts: " << nr_coarse_vertices/double(nv)
         << ", coarse/fine edges: " << coarse_edge_to_vertices.Size()/double(ne)<< endl;

    int smoother_its = 1;
    if (h1_options.variable_vcycle) { smoother_its = pow(2, level_diff); }

    auto h1amg = make_shared<H1AMG_Mat>(sysmat, bjacobi, std::move(prol), smoother_its);

    if (nr_coarse_vertices <= h1_options.min_verts
        || nr_coarse_vertices >= h1_options.vertex_factor * nv
        || h1_options.level <= 1)
      {
        auto sptr_cmat = shared_ptr<BaseSparseMatrix>(coarsemat);
        sptr_cmat->SetInverseType (SPARSECHOLESKY);
        auto inv = sptr_cmat->InverseMatrix();
        h1amg->SetRecursive(inv, sptr_cmat);
      }
    else {
      auto new_options = H1Options(h1_options);
      new_options.level = h1_options.level-1;
      // pass NULL, because all non-free dofs should be collapsed to ground by now
      auto recAMG = BuildAMGMat(coarsemat, coarse_edge_to_vertices, weights_edges_coarse,
                                weights_vertices_coarse, nullptr, new_options);
      h1amg->SetRecursive(recAMG);
    }
    return h1amg;
  }

  unique_ptr<SparseMatrixTM<double>> H1AMG :: CreateProlongation(const Array<int>& vertex_coarse,
                                                                 int ncv, bool complx)
  {
    static Timer Tcreate_prol("H1-AMG::CreateProlongation");
    RegionTimer Rcreate_prol(Tcreate_prol);

    auto nv = vertex_coarse.Size();
    Array<int> non_zero_per_row(nv);
    non_zero_per_row = 0;

    for (auto i = 0; i < nv; ++i) {
      if (vertex_coarse[i] != -1) { non_zero_per_row[i] = 1; }
    }

    unique_ptr<SparseMatrixTM<double>> prol = nullptr;
    if (!complx) {
      prol = unique_ptr<SparseMatrixTM<double>>(new SparseMatrix<double>(non_zero_per_row, ncv));
    } else {
      prol = unique_ptr<SparseMatrixTM<double>>(new SparseMatrix<double, Complex, Complex>(non_zero_per_row, ncv));
    }

    for (auto i = 0; i < nv; ++i) {
      if (vertex_coarse[i] != -1) {
        (*prol)(i, vertex_coarse[i]) = 1;
      }
    }

    return move(prol);
  }

  
  void H1AMG :: ComputeCollapseWeights(const ngstd::Array<INT<2>>& edge_to_vertices,
                                       const ngstd::Array<double>& weights_edges,
                                       const ngstd::Array<double>& weights_vertices,
                                       ngstd::Array<double>& vertex_strength,
                                       ngstd::Array<double>& edge_collapse_weight,
                                       ngstd::Array<double>& vertex_collapse_weight)
  {
    static Timer Tcoll_weights("H1-AMG::ComputeCollapseWeights");
    RegionTimer Rcoll_weights(Tcoll_weights);

    assert(edge_to_vertices.Size() == weights_edges.Size());
    int nr_edges = edge_to_vertices.Size();
    int nr_vertices = weights_vertices.Size();

    vertex_strength.SetSize(nr_vertices);
    vertex_strength = 0.0;
    edge_collapse_weight.SetSize(nr_edges);
    edge_collapse_weight = 0.0;
    vertex_collapse_weight.SetSize(nr_vertices);
    vertex_collapse_weight = 0.0;

    // TODO: Try switching loops to avoid branch misprediction?
    static Timer Tcweights_vertstr("H1-AMG::ComputeCollapseWeights::VertStrength");
    Tcweights_vertstr.Start();
    ParallelFor(nr_edges, [&] (int i) {
        for (int j = 0; j < 2; ++j) {
          AsAtomic(vertex_strength[edge_to_vertices[i][j]]) += weights_edges[i];
        }
      });
    Tcweights_vertstr.Stop();

    static Timer Tcweights_vcollweight("H1-AMG::ComputeCollapseWeights::VertCollWeight");
    Tcweights_vcollweight.Start();
    ParallelFor(nr_vertices, [&] (int i) {
        double current_weight = weights_vertices[i];
        vertex_strength[i] += current_weight;

        // when vertex weight is not 0.0, then also vertex_strength of that vertex
        // can't be 0.0
        if (current_weight != 0.0) {
          vertex_collapse_weight[i] = current_weight / vertex_strength[i];
        }
      });
    Tcweights_vcollweight.Stop();

    static Timer Tcweights_ecollweight("H1-AMG::ComputeCollapseWeights::EdgeCollWeight");
    Tcweights_ecollweight.Start();
    ParallelFor(nr_edges, [&] (int i) {
        double vstr1 = vertex_strength[edge_to_vertices[i][0]];
        double vstr2 = vertex_strength[edge_to_vertices[i][1]];

        // share of the edge weight to the vertex strength
        // same as: weights_edges[i] / vstr1 + weights_edges[i] / vstr2
        edge_collapse_weight[i] = weights_edges[i] * (vstr1+vstr2) / (vstr1 * vstr2);
      });
    Tcweights_ecollweight.Stop();
  }


  template <typename TFUNC>
  void H1AMG :: RunParallelDependency (FlatTable<int> dag,
                                       TFUNC func)
  {
    Array<atomic<int>> cnt_dep(dag.Size());

    for (auto & d : cnt_dep) 
      d.store (0, memory_order_relaxed);

    static Timer t_cntdep("count dep");
    t_cntdep.Start();
    ParallelFor (Range(dag),
                 [&] (int i)
                 {
                   for (int j : dag[i])
                     cnt_dep[j]++;
                 });
    t_cntdep.Stop();    

    atomic<size_t> num_ready(0), num_final(0);
    ParallelForRange (cnt_dep.Size(), [&] (IntRange r)
                      {
                        size_t my_ready = 0, my_final = 0;
                        for (size_t i : r)
                          {
                            if (cnt_dep[i] == 0) my_ready++;
                            if (dag[i].Size() == 0) my_final++;
                          }
                        num_ready += my_ready;
                        num_final += my_final;
                      });

    Array<int> ready(num_ready);
    ready.SetSize0();
    for (int j : Range(cnt_dep))
      if (cnt_dep[j] == 0) ready.Append(j);

    
    if (!task_manager)
      // if (true)
      {
        while (ready.Size())
          {
            int size = ready.Size();
            int nr = ready[size-1];
            ready.SetSize(size-1);
            
            func(nr);
            
            for (int j : dag[nr])
              {
                cnt_dep[j]--;
                if (cnt_dep[j] == 0)
                  ready.Append(j);
              }
          }
        return;
      }

    atomic<int> cnt_final(0);
    SharedLoop2 sl(Range(ready));

    task_manager -> CreateJob 
      ([&] (const TaskInfo & ti)
       {
         size_t my_final = 0;
         TPToken ptoken(queue); 
         TCToken ctoken(queue); 
        
         for (int i : sl)
           queue.enqueue (ptoken, ready[i]);

         while (1)
           {
             if (cnt_final >= num_final) break;

             int nr;
             if(!queue.try_dequeue_from_producer(ptoken, nr)) 
               if(!queue.try_dequeue(ctoken, nr))
                 {
                   if (my_final)
                     {
                       cnt_final += my_final;
                       my_final = 0;
                     }
                   continue;
                 }

             if (dag[nr].Size() == 0)
               my_final++;
             // cnt_final++;

             func(nr);

             for (int j : dag[nr])
               {
                 if (--cnt_dep[j] == 0)
                   queue.enqueue (ptoken, j);
               }
           }
       });
  }


  int H1AMG :: ComputeFineToCoarseVertex(const ngstd::Array<INT<2>>& edge_to_vertices, int nv,
                                         const ngstd::Array<bool>& edge_collapse,
                                         const ngstd::Array<bool>& vertex_collapse,
                                         ngstd::Array<int>& vertex_coarse )
  {
    static Timer Tf2c_verts("H1-AMG::ComputeFineToCoarseVertex");
    RegionTimer Rf2c_verts(Tf2c_verts);

    int nr_edges = edge_to_vertices.Size();
    int nr_coarse_vertices = 0;

    ngstd::Array<int> connected(nv);
    vertex_coarse.SetSize(nv);

    vertex_coarse = -4;

    static Timer Tf2cv_sc("H1-AMG::ComputeFineToCoarseVertex::SelfConnect");
    Tf2cv_sc.Start();
    ParallelFor(nv, [&connected] (int vertex) {
        connected[vertex] = vertex;
      });
    Tf2cv_sc.Stop();

    static Timer Tf2cv_cc("H1-AMG::ComputeFineToCoarseVertex::CoarseConnection");
    Tf2cv_cc.Start();
    // TODO: not sure if we can parallize this
    // Is it possible for more than 1 edge of a vertex to collapse?
    ParallelFor(nr_edges, [&](int edge) {
        if (edge_collapse[edge])
          {
            int vertex1 = edge_to_vertices[edge][0];
            int vertex2 = edge_to_vertices[edge][1];
            if (vertex2>vertex1) {
              AsAtomic(connected[vertex2]) = vertex1;
            }
            else {
              AsAtomic(connected[vertex1]) = vertex2;
            }
          }
      });
    Tf2cv_cc.Stop();

    static Timer Tf2cv_cntcoarse("H1-AMG::ComputeFineToCoarseVertex::CountCoarse");
    Tf2cv_cntcoarse.Start();
    for (int vertex = 0; vertex < nv; ++vertex)
      {
        if (connected[vertex] == vertex)
          {
            if (vertex_collapse[vertex]) {
              vertex_coarse[vertex] = -1;
            }
            else {
              vertex_coarse[vertex] = nr_coarse_vertices++;
            }
          }
      }
    Tf2cv_cntcoarse.Stop();

    // *testout << "vertex_coarse before | after fillup:" << endl;
    static Timer Tf2cv_mapping("H1-AMG::ComputeFineToCoarseVertex::Mapping");
    Tf2cv_mapping.Start();
    ParallelFor(nv, [&connected, &vertex_coarse] (int vertex) {
        if (connected[vertex] != vertex) {
          vertex_coarse[vertex] = vertex_coarse[connected[vertex]];
        }
      });
    Tf2cv_mapping.Stop();

    return nr_coarse_vertices;
  }
  

  void H1AMG :: ComputeFineToCoarseEdge(const ngstd::Array<INT<2>>& edge_to_vertices,
                                        const ngstd::Array<int>& vertex_coarse,
                                        ngstd::Array<int>& edge_coarse,
                                        ngstd::Array<INT<2>>& coarse_edge_to_vertices)
  {
    static Timer t("ComputeFine2CoarseEdge"); RegionTimer reg(t);

    static Timer t1("ComputeFineToCoarseEdge 1");
    static Timer t2("ComputeFineToCoarseEdge 2");
    static Timer t2a("ComputeFineToCoarseEdge 2a");
    static Timer t3("ComputeFineToCoarseEdge 3");

    t1.Start();
    int nr_edges = edge_to_vertices.Size();
    edge_coarse.SetSize(nr_edges);

    ngstd::ParallelHashTable<INT<2>, int> edge_coarse_table;
    // compute fine edge to coarse edge map (edge_coarse)

    ParallelFor (nr_edges, [&] (int edge) 
                 {
                   auto verts = edge_to_vertices[edge];
                   int vertex1 = vertex_coarse[verts[0]];
                   int vertex2 = vertex_coarse[verts[1]];

                   // only edges where both coarse vertices are different and don't
                   // collapse to ground will be coarse edges
                   if (vertex1 != -1 && vertex2 != -1 && vertex1 != vertex2) {
                     edge_coarse_table.Do(INT<2>(vertex1, vertex2).Sort(), [](auto & val) { val = -1; });
                   }
                 });

    t1.Stop();
    t2a.Start();

    Array<int> prefixsums(edge_coarse_table.NumBuckets());
    size_t sum = 0;
    for (size_t i = 0; i < edge_coarse_table.NumBuckets(); i++)
      {
        prefixsums[i] = sum;
        sum += edge_coarse_table.Used(i);
      }
    coarse_edge_to_vertices.SetSize(sum);
    ParallelFor (edge_coarse_table.NumBuckets(),
                 [&] (size_t nr)
                 {
                   int cnt = prefixsums[nr];
                   edge_coarse_table.Bucket(nr).Iterate
                     ([&cnt] (INT<2> key, int & val)
                      {
                        val = cnt++;
                      });
                 });

    t2a.Stop();


    // compute coarse edge to vertex mapping to user for next recursive
    // coarsening
    t2.Start();

    ParallelFor (edge_coarse_table.NumBuckets(),
                 [&] (size_t nr)
                 {               
                   edge_coarse_table.Bucket(nr).Iterate
                     ([&coarse_edge_to_vertices] (INT<2> key, int val)
                      {
                        coarse_edge_to_vertices[val] = key;
                      });
                 });
  
    t2.Stop();

    t3.Start();

    ParallelFor(nr_edges, [&] (int edge)
                {
                  int vertex1 = vertex_coarse[ edge_to_vertices[edge][0] ];
                  int vertex2 = vertex_coarse[ edge_to_vertices[edge][1] ];

                  // only edges where both coarse vertices are different and don't
                  // collapse to ground will be coarse edges
                  if (vertex1 != -1 && vertex2 != -1 && vertex1 != vertex2) {
                    edge_coarse[edge] = edge_coarse_table.Get(INT<2>(vertex1, vertex2).Sort());
                  }
                  else {
                    edge_coarse[edge] = -1;
                  }
                });

    t3.Stop();
  }

  
  void H1AMG :: ComputeCoarseWeightsEdges(const ngstd::Array<INT<2>>& edge_to_vertices,
                                          const ngstd::Array<INT<2>>& coarse_edge_to_vertices,
                                          const ngstd::Array<int>& edge_coarse,
                                          const ngstd::Array<double>& weights_edges,
                                          ngstd::Array<double>& weights_edges_coarse)
  {
    static Timer Tcoarse_eweights("H1-AMG::ComputeCoarseWeightsEdges");
    RegionTimer Rcoarse_eweights(Tcoarse_eweights);

    weights_edges_coarse.SetSize(coarse_edge_to_vertices.Size());
    weights_edges_coarse = 0;

    ParallelFor(edge_to_vertices.Size(), [&] (int i) {
        if (edge_coarse[i] != -1) {
          AsAtomic(weights_edges_coarse[edge_coarse[i]]) += weights_edges[i];
        }
      });
  }


  void H1AMG :: ComputeCoarseWeightsVertices(const ngstd::Array<INT<2>>& edge_to_vertices,
                                             const ngstd::Array<int>& vertex_coarse,
                                             const int nr_coarse_vertices,
                                             const ngstd::Array<double>& weights_edges,
                                             const ngstd::Array<double>& weights_vertices,
                                             ngstd::Array<double>& weights_vertices_coarse)
  {
    static Timer Tcoarse_vweights("H1-AMG::ComputeCoarseWeightsVertices");
    RegionTimer Rcoarse_vweights(Tcoarse_vweights);

    int nr_vertices = weights_vertices.Size();
    int nr_edges = edge_to_vertices.Size();

    *testout << "nrv fine | coarse: " << nr_vertices << " " << nr_coarse_vertices << endl;
    *testout << "nr_edges: " << nr_edges << endl;
    weights_vertices_coarse.SetSize(nr_coarse_vertices);
    weights_vertices_coarse = 0;

    static Timer Tcvweights_fvweight("H1-AMG::ComputeCoarseWeightsVertices::AddFVWeight");
    Tcvweights_fvweight.Start();
    ParallelFor(nr_vertices, [&] (int fine_vertex) {
        int coarse_vertex = vertex_coarse[fine_vertex];
        if (coarse_vertex != -1) {
          AsAtomic(weights_vertices_coarse[coarse_vertex]) += weights_vertices[fine_vertex];
        }
      });
    Tcvweights_fvweight.Stop();

    static Timer Tcvweights_feweight("H1-AMG::ComputeCoarseWeightsVertices::AddFEWeight");
    Tcvweights_feweight.Start();
    ParallelFor(nr_edges, [&] (int fine_edge) {
        for (int i = 0; i < 2; ++i) {
          int cvertex1 = vertex_coarse[ edge_to_vertices[fine_edge][i] ];
          int cvertex2 = vertex_coarse[ edge_to_vertices[fine_edge][1-i] ];
          if (cvertex1 == -1 && cvertex2 != -1) {
            // *testout << "edge " << fine_edge << " between cvert " << cvertex1
            //          << " and " << cvertex2 << endl;
            AsAtomic(weights_vertices_coarse[cvertex2]) += weights_edges[fine_edge];
          }
        }
      });
    Tcvweights_feweight.Stop();
  }

  // Build matrix graph before computing elmats
  shared_ptr<SparseMatrixTM<double>> H1AMG :: H1SmoothedProl(const Array<int>& vertex_coarse, int ncv,
                                                             const Array<INT<2>>& e2v,
                                                             const Array<double>& ew, bool complx)
  {
    int nverts = vertex_coarse.Size();
    int nedges = e2v.Size();

    auto c2f = Coarse2FineVertexTable(vertex_coarse, ncv);
    auto econ = EdgeConnectivityMatrix(e2v, nverts);
    const SparseMatrix<double> & cecon(econ);

    DynamicTable<int> mat_graph(nverts);

    Array<Array<int>> row_dofs(ncv);
    Array<Array<int>> col_dofs(ncv);

    Array<int> c_2_lc(ncv); //coarse to loc coarse
    c_2_lc = -1;
    Matrix<double> m(2);

    Array<int> ext_d(150);
    Array<int> ext_dc(150);
    Array<Array<int> > exd(2);
    exd[0].SetSize(150);
    exd[1].SetSize(150);
    Array<int> all_coarse(50);
    Array<int> ext_e(150);
    Array<Array<int> > v_patch_es(2);
    v_patch_es[0].SetSize(150);
    v_patch_es[1].SetSize(150);
    Array<int> int_e(50);

    Array<double> mat_2_mem(10000);
    Array<double> mat_3_mem(10000);

    Array<INT<2> > carry_over;

    //Build matrix graph
    for (auto patch : Range(ncv)) {

      if (c2f[patch][0] != -1 && c2f[patch][1] != -1) {
        ext_dc.SetSize(0);
        for (auto k : Range(2)) {
          auto rid = econ.GetRowIndices(c2f[patch][k]);
          for(auto d2 : rid) {
            if(c2f[patch][1-k] != d2) {
              if ((!ext_dc.Contains(vertex_coarse[d2])) && (vertex_coarse[d2] != -1)) {
                ext_dc.Append(vertex_coarse[d2]);
              }
            }
          }
        }
        if (ext_dc.Size()==0) {
          for (auto l:Range(2)) {
            if (c2f[patch][1-l]!=-1) {
              mat_graph.Add(c2f[patch][1-l], patch);
              carry_over.Append(INT<2>(c2f[patch][1-l], patch));
            }
          }
        }
        else {
          for (auto d:ext_dc) {
            mat_graph.Add(c2f[patch][0], d);
            mat_graph.Add(c2f[patch][1], d);
          }
          mat_graph.Add(c2f[patch][0], patch);
          mat_graph.Add(c2f[patch][1], patch);

          col_dofs[patch].SetSize(ext_dc.Size()+1);

          for (auto j : Range(ext_dc.Size())) {
            col_dofs[patch][j] = ext_dc[j];
          }

          col_dofs[patch][ext_dc.Size()] = patch;

          row_dofs[patch].SetSize(2);
          row_dofs[patch][0] = c2f[patch][0];
          row_dofs[patch][1] = c2f[patch][1];
        }
      }
      else {
        for (auto l : Range(2)) {
          if (c2f[patch][1-l] != -1) {
            mat_graph.Add(c2f[patch][1-l],patch);
            carry_over.Append(INT<2>(c2f[patch][1-l],patch));
          }
        }
      }
    }


    Array<int> nrd(nverts);
    for (auto k : Range(nverts)) {
      QuickSort(mat_graph[k]);
      nrd[k] = mat_graph[k].Size();
    }

    unique_ptr<SparseMatrixTM<double>> prol = nullptr;
    if (!complx) {
      prol = unique_ptr<SparseMatrixTM<double>>(new SparseMatrix<double>(nrd, ncv));
    } else {
      prol = unique_ptr<SparseMatrixTM<double>>(new SparseMatrix<double, Complex, Complex>(nrd, ncv));
    }

    prol->AsVector() = 0.0;

    for (auto r : Range(nverts)) {
      auto cs = prol->GetRowIndices(r);
      for (auto k : Range(cs.Size())) {
        cs[k] = mat_graph[r][k];
      }
    }

    for (auto t : carry_over) {
      (*prol)(t[0],t[1]) = 1.0;
    }

    //build and add elmats
    for (auto patch : Range(ncv)) {
      if (c2f[patch][0] != -1 && c2f[patch][1] != -1) {
        auto d_patch = c2f[patch];
        int np = d_patch.Size();

        ext_d.SetSize(0);
        ext_dc.SetSize(0);
        exd[0].SetSize(0);
        exd[1].SetSize(0);
        for (auto k : Range(2)) {
          auto d = d_patch[k];
          auto rid = econ.GetRowIndices(d);

          for (auto d2 : rid) {
            if ((!d_patch.Contains(d2)) && (vertex_coarse[d2] != -1))
              {
                ext_d.Append(d2);
                auto vcd2 = vertex_coarse[d2];

                exd[k].Append(vcd2);

                if (!ext_dc.Contains(vcd2)) {
                  ext_dc.Append(vcd2);
                }
              }
          }
        }
        if (ext_dc.Size()!=0) {
          int nexf = ext_d.Size();
          int nexc = ext_dc.Size();

          all_coarse.SetSize(0);
          for (auto d : ext_dc) {
            all_coarse.Append(d);
          }
          all_coarse.Append(patch);

          //no need to sort but also not expensive
          //if you put this back in also put back the other one!
          //QuickSort(all_coarse);
          for (auto k:Range(all_coarse.Size())) {
            c_2_lc[all_coarse[k]] = k;
          }


          int_e.SetSize(0);
          for (auto d : d_patch) {
            for(auto d2 : d_patch) {
              if (d != d2) {
                if (!int_e.Contains(cecon(d,d2))) {
                  int_e.Append(cecon(d,d2));
                }
              }
            }
          }

          ext_e.SetSize(0);
          v_patch_es[0].SetSize(0);
          v_patch_es[1].SetSize(0);
          for (auto k : Range(2))
            {
              auto d = d_patch[k];
              auto ods  = econ.GetRowIndices(d);
              auto enrs = econ.GetRowValues(d);
              for (auto j:Range(ods.Size()) ) {
                if ((!int_e.Contains(enrs[j])) && (vertex_coarse[ods[j]] != -1)) {
                  v_patch_es[k].Append(enrs[j]);
                  if (!ext_e.Contains(enrs[j])) {
                    ext_e.Append(enrs[j]);
                  }
                }
              }
            }
          int nfic = ext_e.Size();

          m = ew[int_e[0]];
          for (auto k : Range(d_patch.Size())) {
            for (auto j : Range(v_patch_es[k].Size())) {
              m(1-k,1-k) += 2*ew[((v_patch_es[k]))[j]];
            }
          }
          m *= 1.0/(m(0,0)*m(1,1)-m(1,0)*m(0,1));

          // np x nfic  nfic x nexc+1
          FlatMatrix<double> m2(d_patch.Size(), nexc+1, &(mat_2_mem[0]));
          m2 = 0.0;

          for (auto k : Range(v_patch_es.Size())) {
            for(auto j : Range(v_patch_es[k].Size())) {
              m2(k,c_2_lc[((exd[k]))[j]]) += ew[((v_patch_es[k]))[j]];
              m2(k,c_2_lc[vertex_coarse[d_patch[0]]]) += ew[((v_patch_es[k]))[j]];
            }
          }
          FlatMatrix<double> m3(d_patch.Size(), nexc+1, &(mat_3_mem[0]));
          m3 = m*m2;

          prol->AddElementMatrix(row_dofs[patch], col_dofs[patch], m3);
        }
      }
    }

    return std::move(prol);
  }


  Table<int> H1AMG :: Coarse2FineVertexTable(const Array<int>& vertex_coarse, int ncv)
  {
    static Timer Tsemi_c2fvt("H1SmoothedProl - Coarse2FineVertexTable");
    RegionTimer Rsemi_c2fvt(Tsemi_c2fvt);

    auto nv = vertex_coarse.Size();

    // two coarse vertex per fine vertex
    Array<int> twos(ncv);
    twos = 2;
    Table<int> c2f(twos);

    // Set all entries of table to -1 initial entry
    // by iterating over rows (FlatArrays can be assigned a scalar value and every entry gets that
    // value)
    for (auto r : c2f) {
      r = -1;
    }

    // invert mapping
    static Timer Tsemi_invmap("H1SmoothedProl - c2fvt - Invert Mapping");
    Tsemi_invmap.Start();
    for (const auto k : Range(nv)) {
      if (vertex_coarse[k] != -1) {
        c2f[vertex_coarse[k]][(c2f[vertex_coarse[k]][0]==-1)?0:1] = k;
      }
    }
    Tsemi_invmap.Stop();

    // sort entries
    static Timer Tsemi_sort("H1SmoothedProl - c2fvt - Sort Entries");
    Tsemi_sort.Start();
    for (auto r : c2f) {
      if (r[0]>r[1]) {
        auto d = r[0];
        r[0] = r[1];
        r[1] = d;
      }
    }
    Tsemi_sort.Stop();

    return c2f;
  }

  // its a square matrix
  // econ(i, j) = #e for which e=(vert_i, vert_j)
  // so rows and cols correspond to vertices and the entry is the edge number
  SparseMatrix<double> H1AMG :: EdgeConnectivityMatrix(const Array<INT<2>>& e2v, int nv)
  {
    static Timer Tsemi_edge_con("H1SmoothedProl - EdgeConnectivityMatrix");
    RegionTimer Rsemi_edge_con(Tsemi_edge_con);
    auto ne = e2v.Size();

    // count nr of connected edges of vertex
    static Timer Tsemi_cnt_con("H1SmoothedProl - edgeconmat- Cnt Connected edges");
    Tsemi_cnt_con.Start();

    Array<int> econ_s(nv);
    econ_s = 0;

    for (auto k : Range(ne)) {
      econ_s[e2v[k][0]]++;
      econ_s[e2v[k][1]]++;
    }
    Tsemi_cnt_con.Stop();

    // build the table for the connections
    static Timer Tsemi_con_table("H1SmoothedProl - edgeconmat - Connection table");
    Tsemi_con_table.Start();
    Table<int> tab_econ(econ_s);
    econ_s = 0; // used again for counting!!!!
    for (auto k : Range(ne)) {
      tab_econ[e2v[k][0]][econ_s[e2v[k][0]]++] = k;
      tab_econ[e2v[k][1]][econ_s[e2v[k][1]]++] = k;
    }
    Tsemi_con_table.Stop();

    // build the edge connection matrix
    static Timer Tsemi_con_matrix("H1SmoothedProl - edgeconmat - create matrix");
    Tsemi_con_matrix.Start();
    SparseMatrix<double> econ(econ_s, nv);
    econ.AsVector() = -1;
    for (auto k:Range(nv)) {
      auto ind = econ.GetRowIndices(k);
      auto val = econ.GetRowValues(k);
      Array<int> ods(ind.Size());
      int cnt = 0;

      for (auto j:Range(econ_s[k])) {
        auto enr = tab_econ[k][j];
        auto d = (e2v[enr][0]!=k)?e2v[enr][0]:e2v[enr][1];
        ods[cnt++] = d;
      }

      Array<int> indices(ind.Size());
      for (auto k : Range(ind.Size())) { indices[k]=k; }
      QuickSortI(ods,indices);

      for (auto j : Range(econ_s[k])) {
        ind[j] = ods[indices[j]];
        val[j] = tab_econ[k][indices[j]];
      }
    }
    Tsemi_con_matrix.Stop();

    // Sanity checks
    static Timer Tsemi_con_checks("H1SmoothedProl - edgeconmat - Checks");
    Tsemi_con_checks.Start();
    for (auto k : Range(econ.Height())) {
      auto vs = econ.GetRowValues(k);
      for (auto d : econ.GetRowIndices(k)) {
        if (d<0) {
          cout << endl << endl << "STOOOOOOP2" << endl << endl;
        }
      }
      for (auto j : Range(vs.Size())) {
        if (vs[j]==-1) {
          cout << endl << endl << "STOOOOOOOP" << endl << endl;
        }
      }
    }
    Tsemi_con_checks.Stop();

    return econ;
  }


  shared_ptr<SparseMatrixTM<double>> H1AMG :: CreateSmoothedProlongation
  (const Array<INT<2>>& e2v, const Array<double>& eweights, int nv,
   const unique_ptr<SparseMatrixTM<double>> triv_prol)
  {
    static Timer Tsmoothed("H1 Create Smoothed Prolongation");
    RegionTimer Rsmoothed(Tsmoothed);

    unique_ptr<SparseMatrixTM<double>> subst = BuildOffDiagSubstitutionMatrix(e2v, eweights, nv);
    cout << "Smoothed prol start" << endl;

    static Timer Tsmoothed_jacobi("H1 Create Smoothed Prol - Jacobi Mat");
    Tsmoothed_jacobi.Start();
    double avg = 0.5;
    ParallelFor (nv, [&] (int vert) {
        auto row_indices = subst->GetRowIndices(vert);
        auto row_vals = subst->GetRowValues(vert);
        double sum = 0.;
        for (auto v : row_vals) {
          sum += v;
        }
        for (auto i : row_indices) {
          // minus sign is in sum because we summed over negatives
          (*subst)(vert, i) *= avg/sum;
        }
        (*subst)(vert, vert) += 1. - avg;
      });
    Tsmoothed_jacobi.Stop();

    assert(subst->Width() == triv_prol->Height());
    shared_ptr<SparseMatrixTM<double>> smoothed_prol = MatMult(*subst, *triv_prol);
  
    return std::move(smoothed_prol);
  }

  unique_ptr<SparseMatrixTM<double>> H1AMG :: BuildOffDiagSubstitutionMatrix
  (const Array<INT<2>>& e2v, const Array<double>& eweights, int nv)
  {
    static Timer Tsubst("H1 Build Subst Matrix");
    RegionTimer Rsubst(Tsubst);

    auto ne = e2v.Size();

    static Timer Tsubst_nze("H1 Build Subst Matrix - count nze");
    Tsubst_nze.Start();
    Array<int> nze(nv);
    ParallelFor(nv, [&] (int vert) {
        nze[vert] = 1;
      });

    ParallelFor(ne, [&] (int edge) {
        AsAtomic(nze[e2v[edge][0]])++;
        AsAtomic(nze[e2v[edge][1]])++;
      });

    // TODO: Check for 2d or 3d problem to decide maximum number of nze per row.
    // 1 entry for the corresponding coarse vertex plus 2d: 3, 3d: 4
    int max_nze_per_row = 5;  // use this for 3d. 4 + 1
    ParallelFor(nv, [&] (int vert) {
        nze[vert] = min(nze[vert], max_nze_per_row);
      });
    Tsubst_nze.Stop();

    static Timer Tsubst_mem("H1 Build Subst Matrix - memory");
    Tsubst_mem.Start();
    auto subst = std::make_unique<ngla::SparseMatrix<double>>(nze, nv);
    Tsubst_mem.Stop();

    static Timer Tsubst_table("H1 Build Subst Matrix - build table");
    Tsubst_table.Start();
    TableCreator<pair<int, double>> creator(nv);
    for (; !creator.Done(); creator++) {
      ParallelFor(nv, [&] (auto vert) {
          creator.Add(vert, make_pair<>(vert, 0));
        });
      ParallelFor(ne, [&] (auto edge) {
          auto ew = eweights[edge];
          auto v1 = e2v[edge][0];
          auto v2 = e2v[edge][1];

          creator.Add(v1, make_pair<>(v2, -ew));
          creator.Add(v2, make_pair<>(v1, -ew));
        });
    }
    auto table = creator.MoveTable();
    Tsubst_table.Stop();

    // row indices for sparse matrix need to be sorted
    static Timer Tsubst_sort("H1 Build Subst Matrix - sort rows in table");
    Tsubst_sort.Start();
    auto less_second = [] (pair<int, double> lhs, pair<int, double> rhs) {
      return lhs.second < rhs.second;
    };
ParallelFor(table.Size(), [&] (auto row) {
      QuickSort(table[row], less_second);
    }, TasksPerThread(5));
  Tsubst_sort.Stop();

  static Timer Tsubst_write("H1 Build Subst Matrix - write into matrix");
  Tsubst_write.Start();
  auto less_first = [] (pair<int, double> lhs, pair<int, double> rhs) {
    return lhs.first < rhs.first;
  };
  ParallelFor(table.Size(), [&] (auto row_i) {
      auto row = table[row_i];
      auto row_ind = subst->GetRowIndices(row_i);
      auto row_vals = subst->GetRowValues(row_i);
      ArrayMem<pair<int, double>, 5> new_row(row_ind.Size());
      new_row[0].first = row_i;
      new_row[0].second = 0;
      for (int i = 1; i < new_row.Size(); ++i) {
        new_row[i].first = row[i-1].first;
        new_row[i].second = row[i-1].second;
      }
      QuickSort(new_row, less_first);
      for (int i = 0; i < row_ind.Size(); ++i) {
        row_ind[i] = new_row[i].first;
        row_vals[i] = new_row[i].second;
      }
    });
  Tsubst_write.Stop();

  return std::move(subst);
}
  double H1AMG :: SplitMatrixConstant(ngbla::FlatMatrix<double> source_matrix,
                                      ngbla::FlatMatrix<double>& constant_matrix,
                                      ngbla::FlatMatrix<double>& nullspace_matrix)
  {
    const int nr_dofs = source_matrix.Height();

    double sum_matrix = 0;
    for (auto val : source_matrix.AsVector()) {
      sum_matrix += val;
    }
    double c = sum_matrix / (nr_dofs * nr_dofs);

    const double TOL = 1e-10;
    if (abs(c) <= TOL) c = 0;

    constant_matrix = c;
    nullspace_matrix = source_matrix - constant_matrix;

    return c;
  }

  RegisterPreconditioner<H1AMG> initmyamg("h1amg");

}  // h1amg
