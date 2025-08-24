/*
 * dsmc_base.h
 *
 *  Created on: Jun 1, 2010
 *      Author: denis
 */

#ifndef DSMC_BASE_H_
#define DSMC_BASE_H_

#include <ostream>
#include <string>

#include "draw_helper.h"
#include "model.h"
#include "sim_settings.h"
#include "statistics.h"

namespace dsmc {
class DSMCSolver {
 public:
  DSMCSolver();
  virtual ~DSMCSolver();

  void Init(const settings_t& s);
  void DoBenchmark(uint itn, float dt);

  virtual void RunSimulationStep(float dt) = 0;
  virtual void Render() = 0;
  virtual uint InitSystemState(const float3& streamVel) = 0;

  void RegisterSimState(simulation_state_t* state);
  void DumpStatistics(std::ostream& out, bool csv = false);
  void SetModel(Model* model);

  virtual void RenderConcMap(uint x, uint y, uint w, uint h) = 0;

  void RenderVectorField();

  void SaveColorMap(const std::string& filename);
  void SaveVelocityField(const std::string& filename);

 protected:
  virtual void RunBenchmarkSimulationStep(float dt);

  void ProcessAndOutputStatistics(statistics_t* stat, float4 collStat,
                                  std::ostream& out, bool printCsv);

  virtual statistics_t* SampleStatistics(float4& collStat) = 0;
  virtual void InitData() = 0;
  virtual void InitRegularGrid(reg_grid_t& grid, const float* v, uint size) = 0;

  virtual unsigned char* GetColorMap(uint& w, uint& h, uint& bpp) = 0;
  virtual float3* GetVelocityField() = 0;

  settings_t m_settings;
  simulation_state_t* m_state;
  uint vectorFieldBuffer;
};

enum SolverType { SolverGPU, SolverGPUVBO, SolverCPU };

DSMCSolver* CreateSolver(SolverType);

}  // namespace dsmc

#endif /* DSMC_BASE_H_ */
