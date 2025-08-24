#ifndef RNDFAST_H
#define RNDFAST_H

namespace cudahelp {
namespace rand {

bool InitFastRngs(int count);
void DeinitFastRngs();

}  // namespace rand
}  // namespace cudahelp

#endif