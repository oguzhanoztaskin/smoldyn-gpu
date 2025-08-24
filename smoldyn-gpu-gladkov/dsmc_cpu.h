#ifndef DSMC_CPU_H
#define DSMC_CPU_H

void InitGridCPU(unsigned int numCells, unsigned int numParticles);
void ComputeNanbuCPU(float* d_speeds, unsigned int bin_size);
void GenerateInitialSpeedsCPU(float* vels, float* pos, float sx, float sy,
                              unsigned int count);
void GenerateInitialSpeedsPerCellCPU(float*, float*, float sx, float sy,
                                     unsigned int partPerCell,
                                     unsigned int bin_size);
void ComputeNanbuBabovskyCPU(float* vels, unsigned int bin_size);
void ComputeBirdCPU(float* vels, float* d_pos, float* d_model, int vert_count,
                    unsigned int bin_size, unsigned int count);

void ResetPositionsCPU(float* d_pos, unsigned int count);
void ProcessModelCPU(float* d_model, int vert_count);
void IntegrateCPU(float* d_vels, float* d_pos, float* d_cols, float dt,
                  unsigned int count);
void UpdateGridCPU(float* d_pos, unsigned int count);

#endif
