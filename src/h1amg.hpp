#ifndef H1AMG_HPP_
#define H1AMG_HPP_

#include <memory>
#include <mutex>

#include <comp.hpp>
#include <la.hpp>

#include "h1.hpp"

namespace h1amg
{
  using namespace ngla;

class H1AMG : public ngcomp::Preconditioner
{
public:
  struct H1Options
  {
    int maxlevel = 10;
    int level = 10;
    bool variable_vcycle = false;
    float vertex_factor = 0.8;
    int min_verts = 1000;
    bool smoothed = false;
    bool semi_smoothed = true;
    int special_level = 3;
  };

private:
  shared_ptr<ngcomp::BilinearForm> bfa;

  shared_ptr<H1AMG_Mat> amg_matrix;

  size_t m_ndof;
  ngstd::HashTable<INT<2>,double> dof_pair_weights;
  ngstd::ParallelHashTable<INT<2>,double> par_dof_pair_weights;
  ngstd::Array<double> weights_vertices;

  ngstd::Array<std::mutex> m_hashlocks;

  shared_ptr<ngla::BitArray> freedofs;

  H1Options m_h1_options;

public:
  H1AMG(
      const ngcomp::PDE& a_pde, const ngstd::Flags& a_flags,
      const string a_name = "H1AMG Preconditioner");

  H1AMG(
      shared_ptr<ngcomp::BilinearForm> a_bfa, const Flags& a_flags,
      const string a_name = "H1AMG Preconditioner");

  virtual ~H1AMG() override
  { }

  virtual void Update() override
  { }

  virtual const ngla::BaseMatrix& GetAMatrix() const override
  { return bfa->GetMatrix(); }

  virtual const ngla::BaseMatrix& GetMatrix() const override
  { return *amg_matrix; }

  virtual void InitLevel(shared_ptr<BitArray> afreedofs = nullptr) override;

  virtual void FinalizeLevel(const BaseMatrix* mat) override;

  void AddElementMatrixCommon(
      ngstd::FlatArray<int> dnums, const ngbla::FlatMatrix<double>& elmat, ngstd::LocalHeap& lh);

  virtual void AddElementMatrix(
      ngstd::FlatArray<int> dnums, const ngbla::FlatMatrix<double>& elmat, ngcomp::ElementId ei,
      ngstd::LocalHeap& lh) override;

  virtual void AddElementMatrix(
      ngstd::FlatArray<int> dnums, const ngbla::FlatMatrix<Complex>& elmat, ngcomp::ElementId ei,
      ngstd::LocalHeap& lh) override;

  virtual const char* ClassName() const override
  { return "H1AMG Preconditioner"; }

  using Vertex = size_t;
  struct Edge {
    explicit Edge(std::size_t a_id, Vertex a_v1, Vertex a_v2)
      : id(a_id), v1(a_v1), v2(a_v2)
    { }

    std::size_t id;
    Vertex v1;
    Vertex v2;
  };

private:
  shared_ptr<H1AMG_Mat> BuildAMGMat(shared_ptr<SparseMatrixTM<double>> sysmat,
                                    const Array<INT<2>>& edge_to_vertices,
                                    const Array<double>& weights_edges,
                                    const Array<double>& weights_vertices,
                                    shared_ptr<BitArray> free_dofs, const H1Options& h1_options);

  unique_ptr<SparseMatrixTM<double>> CreateProlongation(const Array<int>& vertex_coarse,
                                                        int ncv, bool complx);

  // Compute the balanced collapse weights for edges and vertices from the
  // given weights.
  // All collapse weights added up should amount to the number of vertices.
  void ComputeCollapseWeights(const ngstd::Array<INT<2>>& edge_to_vertices,
                              const ngstd::Array<double>& weights_edges,
                              const ngstd::Array<double>& weights_vertices,
                              ngstd::Array<double>& vertex_strength,
                              ngstd::Array<double>& edge_collapse_weight,
                              ngstd::Array<double>& vertex_collapse_weight);

  // Compute how fine vertices get mapped to coarse vertices, given which edges
  // and vertices to collapse.
  // -1 in vertex_coarse means vertex maps to ground.
  // Returns number of coarse vertices.
  int ComputeFineToCoarseVertex(const ngstd::Array<INT<2>>& edge_to_vertices, int nverts,
                                const ngstd::Array<bool>& edge_collapse,
                                const ngstd::Array<bool>& vertex_collapse,
                                ngstd::Array<int>& vertex_coarse);


  // Compute how fine edges map to coarse edges, using the alreade computed
  // vertex mapping.
  void ComputeFineToCoarseEdge(const ngstd::Array<INT<2>>& edge_to_vertices,
                               const ngstd::Array<int>& vertex_coarse,
                               ngstd::Array<int>& edge_coarse,
                               ngstd::Array<INT<2>>& coarse_edge_to_vertices);

  // Computes coarse edge weights to use for the next iteration.
  void ComputeCoarseWeightsEdges(const ngstd::Array<INT<2>>& edge_to_vertices,
                                 const ngstd::Array<INT<2>>& coarse_edge_to_vertices,
                                 const ngstd::Array<int>& edge_coarse,
                                 const ngstd::Array<double>& weights_edges,
                                 ngstd::Array<double>& weights_edges_coarse);

  // Compute coarse vertex weights to use for the next iteration.
  void ComputeCoarseWeightsVertices(const ngstd::Array<INT<2>>& edge_to_vertices,
                                    const ngstd::Array<int>& vertex_coarse,
                                    const int nr_coarse_vertices,
                                    const ngstd::Array<double>& weights_edges,
                                    const ngstd::Array<double>& weights_vertices,
                                    ngstd::Array<double>& weights_vertices_coarse);

  shared_ptr<SparseMatrixTM<double>> H1SmoothedProl(const ngstd::Array<int>& vertex_coarse, int ncv,
                                                    const ngstd::Array<ngstd::INT<2>>& e2v,
                                                    const ngstd::Array<double>& ew, bool complx=false);

  shared_ptr<SparseMatrixTM<double>> CreateSmoothedProlongation
  (const ngstd::Array<ngstd::INT<2>>& e2v, const ngstd::Array<double>& eweights,
   int nv, const unique_ptr<SparseMatrixTM<double>> triv_prol);

  Table<int> Coarse2FineVertexTable(const Array<int>& vertex_coarse, int ncv);
  SparseMatrix<double> EdgeConnectivityMatrix(const Array<INT<2>>& e2v, int nv);

  unique_ptr<SparseMatrixTM<double>> BuildOffDiagSubstitutionMatrix(const Array<INT<2>>& e2v,
                                                                    const Array<double>& eweights,
                                                                    int nv);
  // Splits given matrix in one with constant functions in nullspace
  // and a constant part.
  // Returns a constant, such that constant_matrix = c * one_matrix
  double SplitMatrixConstant(ngbla::FlatMatrix<double> source_matrix,
                             ngbla::FlatMatrix<double>& constant_matrix,
                             ngbla::FlatMatrix<double>& nullspace_matrix);

  template <typename TFUNC>
  void RunParallelDependency (FlatTable<int> dag, TFUNC func);
};

  inline bool operator<(const H1AMG::Edge& lhs, const H1AMG::Edge& rhs)
  { return lhs.id < rhs.id; }

  inline bool operator==(const H1AMG::Edge& lhs, const H1AMG::Edge& rhs)
  { return (lhs.id == rhs.id) && (lhs.v1 == rhs.v1) && (lhs.v2 == rhs.v2); }

  inline std::ostream& operator<<(std::ostream& str, const H1AMG::Edge& edge)
  {
    str << "Edge " << edge.id << ": (" << edge.v1 << ", " << edge.v2 << ")";
    return str;
  }


}  // h1amg

#endif  // H1AMG_HPP_
