#ifndef MTWISTER_H
#define MTWISTER_H

namespace cudahelp {
namespace rand {

bool InitGPUTwisters(const char* fname, int rngsCount, unsigned int seed);
void DeinitGPUTwisters();

}  // namespace rand
}  // namespace cudahelp

#endif
