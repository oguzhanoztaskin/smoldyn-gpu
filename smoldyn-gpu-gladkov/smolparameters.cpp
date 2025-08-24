/*
 * smolparameters.cpp
 *
 *  Created on: Nov 3, 2010
 *      Author: denis
 */

#include "smolparameters.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include "file_path.h"

double unbindingradius(double pgem, double dt, double difc, double a);
double bindingradius(double rate, double dt, double difc, double b, int rel);

namespace smolgpu {
/*
struct	species_t
{
        std::string	name;
        int		id;
        float	difc;
        float3	color;
        int		count;
        float3	pos;
        bool	posRandom[3];
};

class	smolparams_t
{
public:
        bool	useGraphics;
        int		dim;

        int		max_mol;
        typedef	std::map<std::string,species_t>	species_map_t;
        species_map_t	species;

        bbox_t	boundaries;
        char	boundCond[3];
        float	startTime;
        float	endTime;
        float	timeStep;
};

struct	firstreact_t: public	zeroreact_t
{
        int		reactant;
        float	tau;

        void	calcProb(float dt, float vol);

        void	dump(std::ostream&	ostr) const;
};
*/

int smolparams_t::getSpieceID(const std::string& name) const {
  smolparams_t::species_map_t::const_iterator it = species.find(name);

  return (it == species.end()) ? -1 : it->second.id;
}

int smolparams_t::getActualMolsNum() const {
  smolparams_t::species_map_t::const_iterator it = species.begin();
  smolparams_t::species_map_t::const_iterator end = species.end();

  uint count = 0;

  while (it != end) {
    count += it->second.getNumberOfMols();
    ++it;
  }

  return count;
}

int smolparams_t::getGridDim() const {
  return pow(getActualMolsNum() / 4.0f, 1.0f / 3.0f) + 0.5f;
}

const species_t& smolparams_t::getSpiece(int id) const {
  smolparams_t::species_map_t::const_iterator it = species.begin();
  smolparams_t::species_map_t::const_iterator end = species.end();

  static species_t zero;
  zero.id = -1;

  while (it != end) {
    if (it->second.id == id) return it->second;
    ++it;
  }

  return zero;
}

void zeroreact_t::calcProb(float dt, float vol) { prob = rate * dt * vol; }

void smolparams_t::calcReactProb() {
  smolparams_t::zeroreact_map_t::iterator it = zeroreact.begin();
  smolparams_t::zeroreact_map_t::iterator end = zeroreact.end();

  float vol = boundaries.get_vol();

  while (it != end) {
    it->second.calcProb(timeStep, vol);
    ++it;
  }

  calcFirstOrderReactProb();
}
/*
 * 			for(j=0;j<rxnss->nrxn[i];j++) {
                                rxn2=rxnss->rxn[rxnss->table[i][j]];
                                if(rxn2==rxn) j=rxnss->nrxn[i];
                                else sum2+=rxn2->prob; }
                        rxn->prob=rxn->prob/(1.0-sum2); }		//
 probability, accounting for prior reactions
 *
 */

// TODO: change map to vector to keep order of reactions (as they appear in the
// file)
void smolparams_t::calcFirstOrderReactProb() {
  std::vector<float> ratesSum(species.size(), 0);

  smolparams_t::firstreact_map_t::iterator it = firstreact.begin();
  smolparams_t::firstreact_map_t::iterator end = firstreact.end();

  for (; it != end; ++it) {
    if (it->second.rate > 0) it->second.tau = 1.0f / it->second.rate;

    ratesSum[it->second.reactant] += it->second.rate;
  }

  it = firstreact.begin();

  for (; it != end; ++it) {
    float sum = ratesSum[it->second.reactant];
    if (sum > 0)
      it->second.prob = it->second.rate / sum * (1.0f - exp(-timeStep * sum));

    smolparams_t::firstreact_map_t::iterator b = firstreact.begin();

    sum = 0;

    for (; b != it; b++)
      if (b->second.reactant == it->second.reactant) sum += b->second.prob;

    it->second.prob = it->second.prob / (1 - sum);
  }
}

void smolparams_t::calcSecondOrderReactData() {
  smolparams_t::secondreact_map_t::iterator it = secondreact.begin();
  smolparams_t::secondreact_map_t::iterator end = secondreact.end();

  for (; it != end; ++it) {
    secondreact_t& rc = it->second;

    rc.unbindrad = 0;
    rc.prob = 1.0f;

    if (rc.bindrad2 > 0.0f) continue;

    float rt = rc.rate;

    if (rc.reactant == rc.reactant2) rt *= 2;

    float dsum = getSpiece(rc.reactant).difc + getSpiece(rc.reactant2).difc;

    rc.bindrad2 = bindingradius(rt, timeStep, dsum, -1, 0);
    rc.bindrad2 *= rc.bindrad2;

    //		rc.prob = 1.0-exp(-timeStep*rc.rate);
  }
}

void smolparams_t::calcParameters()  // const std::string& path, const
                                     // std::string& prefix)
{
  uint d = getGridDim();

  csize.x = (boundaries.max.x - boundaries.min.x) / d;
  csize.y = (boundaries.max.y - boundaries.min.y) / d;
  csize.z = (boundaries.max.z - boundaries.min.z) / d;
  /*
          if(prefix != "" || path != "")
                  for(int i = 0; i < cmds.size(); i++)
                          cmds[i].process_filename(path, prefix);
  */
  calcReactProb();
  calcSecondOrderReactData();
}

void smolparams_t::dump(std::ostream& ostr) const {
  ostr << "Use graphics: " << useGraphics << "\nDim: " << dim
       << "\nMaxMol: " << max_mol << "\n";
  ostr << "Boundaries min: " << boundaries.min.x << "," << boundaries.min.y
       << "," << boundaries.min.z << "\n";
  ostr << "Boundaries max: " << boundaries.max.x << "," << boundaries.max.y
       << "," << boundaries.max.z << "\n";
  ostr << "Boundary conditions: " << boundCond[0] << "," << boundCond[1] << ","
       << boundCond[2] << "\n";
  ostr << "Time min: " << startTime << " Time max: " << endTime
       << " Timestep: " << timeStep << "\n";

  int gd = getGridDim();

  ostr << "Grid dim is: " << gd << "x" << gd << "x" << gd << "\n";
  ostr << "Molecules per cell: " << (float)getActualMolsNum() / gd / gd / gd
       << "\n";
  ostr << "Cell size is: " << csize.x << "x" << csize.y << "x" << csize.z
       << "\n";
  ostr << "Dumping molmoments: " << dumpMolMoments
       << " molcounts: " << dumpMolCounts << "\n";

  ostr << "Accuracy is: " << accuracy << "\n";

  ostr << "Species:\n";

  smolparams_t::species_map_t::const_iterator it = species.begin();
  smolparams_t::species_map_t::const_iterator end = species.end();

  while (it != end) {
    it->second.dump(ostr);
    ++it;
  }

  ostr << "Reactions (0th order): \n";

  smolparams_t::zeroreact_map_t::const_iterator it1 = zeroreact.begin();
  smolparams_t::zeroreact_map_t::const_iterator end1 = zeroreact.end();

  while (it1 != end1) {
    it1->second.dump(ostr, *this);
    ++it1;
  }

  ostr << "Reactions (1th order): \n";

  smolparams_t::firstreact_map_t::const_iterator it2 = firstreact.begin();
  smolparams_t::firstreact_map_t::const_iterator end2 = firstreact.end();

  while (it2 != end2) {
    it2->second.dump(ostr, *this);
    ++it2;
  }

  ostr << "Reactions (2th order): \n";

  smolparams_t::secondreact_map_t::const_iterator it3 = secondreact.begin();
  smolparams_t::secondreact_map_t::const_iterator end3 = secondreact.end();

  while (it3 != end3) {
    it3->second.dump(ostr, *this);
    ++it3;
  }

  ostr << "Commands: \n";

  for (int i = 0; i < cmds.size(); i++) cmds[i].dump(ostr, *this);
}

void species_t::dump(std::ostream& ostr) const {
  ostr << name << "\nId: " << id << "\nDifc: " << difc << "\nColor: " << color.x
       << "," << color.y << "," << color.z << "\n";
  /*
          for(int i = 0; i < distr.size(); i++)
          {
                  ostr<<"Set: "<<i<<"\n";
                  ostr<<"Count: "<<distr[i].count<<"\nPos:
  "<<distr[i].pos.x<<","<<distr[i].pos.y<<","<<distr[i].pos.z<<"\n";
                  ostr<<"Random pos:
  "<<distr[i].posRandom[0]<<","<<distr[i].posRandom[1]<<","<<distr[i].posRandom[2]<<"\n";
  //		ostr<<"Dumping to: "<<out_file<<"\n";
          }
          */
}
/*
        std::string	name;
        std::vector<int>	products;
        float	rate;
        float	prob;
*/
void zeroreact_t::dump(std::ostream& ostr, const smolparams_t& p) const {
  ostr << name << "\nProducts:\n";
  for (int i = 0; i < products.size(); i++)
    ostr << "Id: " << products[i] << " Name: " << p.getSpiece(products[i]).name
         << "\n";
  ostr << "Rate: " << rate << " Probability: " << prob << "\n";
}

void firstreact_t::calcProb(float dt, float vol) {
  if (rate > 0) tau = 1 / rate;
}

bool firstreact_t::deallocMolecules() const {
  for (int i = 0; i < products.size(); i++)
    if (products[i] == -1) return true;
}

void firstreact_t::dump(std::ostream& ostr, const smolparams_t& p) const {
  zeroreact_t::dump(ostr, p);

  ostr << "Reactant Id: " << reactant << " Name: " << p.getSpiece(reactant).name
       << " Tau: " << tau << "\n";
}

void secondreact_t::dump(std::ostream& ostr, const smolparams_t& p) const {
  firstreact_t::dump(ostr, p);

  ostr << "Second reactant Id: " << reactant2
       << " Name: " << p.getSpiece(reactant2).name
       << " Binding radius: " << bindrad2 << "\n";
}

}  // namespace smolgpu

#define PI 3.14159265358979323846
#define SQRT2PI 2.50662827462

/* numrxnrate calculates the bimolecular reaction rate for the simulation
algorithm, based on the rms step length in step, the binding radius in a, and
the unbinding radius in b.  It uses cubic polynomial interpolation on previously
computed data, with interpolation both on the reduced step length (step/a) and
on the reduced unbinding radius (b/a).  Enter a negative value of b for
irreversible reactions.  If the input parameters result in reduced values that
are off the edge of the tabulated data, analytical results are used where
possible.  If there are no analytical results, the data are extrapolated.
Variable components: s is step, i is irreversible, r is reversible with b>a, b
is reversible with b<a, y is a rate.  The returned value is the reaction rate
times the time step delta_t.  In other words, this rate constant is per time
step, not per time unit; it has units of length cubed.  */
double numrxnrate(double step, double a, double b) {
  static const double yi[] = {
      // 600 pts, eps=1e-5, up&down, error<2.4%
      4.188584,  4.188381,  4.187971, 4.187141,  4.185479,  4.182243,
      4.1761835, 4.1652925, 4.146168, 4.112737,  4.054173,  3.95406,
      3.7899875, 3.536844,  3.178271, 2.7239775, 2.2178055, 1.7220855,
      1.2875175, 0.936562,  0.668167, 0.470021,  0.327102,  0.225858,
      0.1551475, 0.1061375, 0.072355, 0.049183,  0.0333535, 0.022576,
      0.015257};
  static const double yr[] = {
      // 500 pts, eps=1e-5, up&down, error<2.2%
      4.188790, 4.188790, 4.188790, 4.188788, 4.188785, 4.188775, 4.188747,
      4.188665, 4.188438, 4.187836, 4.186351, 4.182961, 4.175522, 4.158666,
      4.119598, 4.036667, 3.883389, 3.641076, 3.316941, 2.945750, 2.566064,
      2.203901, 1.872874, 1.578674, 1.322500, 1.103003, 0.916821, 0.759962,
      0.628537, 0.518952, 0.428136, 4.188790, 4.188790, 4.188789, 4.188786,
      4.188778, 4.188756, 4.188692, 4.188509, 4.188004, 4.186693, 4.183529,
      4.176437, 4.160926, 4.125854, 4.047256, 3.889954, 3.621782, 3.237397,
      2.773578, 2.289205, 1.831668, 1.427492, 1.087417, 0.811891, 0.595737,
      0.430950, 0.308026, 0.217994, 0.152981, 0.106569, 0.073768, 4.188790,
      4.188789, 4.188788, 4.188783, 4.188768, 4.188727, 4.188608, 4.188272,
      4.187364, 4.185053, 4.179623, 4.167651, 4.141470, 4.082687, 3.956817,
      3.722364, 3.355690, 2.877999, 2.353690, 1.850847, 1.411394, 1.050895,
      0.768036, 0.553077, 0.393496, 0.277134, 0.193535, 0.134203, 0.092515,
      0.063465, 0.043358, 4.188790, 4.188789, 4.188786, 4.188777, 4.188753,
      4.188682, 4.188480, 4.187919, 4.186431, 4.182757, 4.174363, 4.156118,
      4.116203, 4.028391, 3.851515, 3.546916, 3.110183, 2.587884, 2.055578,
      1.574898, 1.174608, 0.858327, 0.617223, 0.438159, 0.307824, 0.214440,
      0.148358, 0.102061, 0.069885, 0.047671, 0.032414, 4.188789, 4.188788,
      4.188783, 4.188769, 4.188729, 4.188614, 4.188288, 4.187399, 4.185108,
      4.179638, 4.167499, 4.141379, 4.084571, 3.964576, 3.739474, 3.381771,
      2.906433, 2.372992, 1.854444, 1.401333, 1.032632, 0.746509, 0.531766,
      0.374442, 0.261247, 0.180929, 0.124557, 0.085332, 0.058229, 0.039604,
      0.026865, 4.188789, 4.188786, 4.188778, 4.188756, 4.188692, 4.188510,
      4.188002, 4.186650, 4.183288, 4.175566, 4.158864, 4.123229, 4.047257,
      3.896024, 3.632545, 3.242239, 2.750962, 2.220229, 1.717239, 1.285681,
      0.939696, 0.674540, 0.477621, 0.334612, 0.232460, 0.160415, 0.110103,
      0.075242, 0.051236, 0.034789, 0.023565, 4.188788, 4.188784, 4.188772,
      4.188737, 4.188637, 4.188354, 4.187585, 4.185607, 4.180895, 4.170490,
      4.148460, 4.102035, 4.006907, 3.829897, 3.541151, 3.133335, 2.635739,
      2.109597, 1.619284, 1.204298, 0.875185, 0.625190, 0.440882, 0.307818,
      0.213235, 0.146800, 0.100560, 0.068608, 0.046655, 0.031645, 0.021415,
      4.188787, 4.188780, 4.188761, 4.188707, 4.188552, 4.188124, 4.186995,
      4.184222, 4.177931, 4.164527, 4.136619, 4.079198, 3.967945, 3.772932,
      3.468711, 3.050092, 2.548726, 2.026932, 1.547039, 1.144954, 0.828592,
      0.589866, 0.414774, 0.288895, 0.199728, 0.137273, 0.093903, 0.063994,
      0.043478, 0.029467, 0.019931, 4.188785, 4.188774, 4.188745, 4.188661,
      4.188426, 4.187795, 4.186203, 4.182494, 4.174471, 4.157859, 4.123939,
      4.056910, 3.934028, 3.727335, 3.412145, 2.985387, 2.481620, 1.963935,
      1.492519, 1.100533, 0.793976, 0.563797, 0.395615, 0.275071, 0.189896,
      0.130359, 0.089086, 0.060661, 0.041187, 0.027899, 0.018863, 4.188782,
      4.188766, 4.188720, 4.188592, 4.188244, 4.187347, 4.185205, 4.180486,
      4.170691, 4.150875, 4.111574, 4.037574, 3.906673, 3.691217, 3.367270,
      2.934386, 2.429282, 1.915212, 1.450660, 1.066615, 0.767733, 0.544143,
      0.381225, 0.264724, 0.182558, 0.125209, 0.085504, 0.058187, 0.039490,
      0.026741, 0.018074, 4.188777, 4.188752, 4.188683, 4.188492, 4.187993,
      4.186777, 4.184049, 4.178342, 4.166861, 4.144167, 4.100756, 4.021817,
      3.884852, 3.662169, 3.331386, 2.893934, 2.388069, 1.877097, 1.418051,
      1.040351, 0.747549, 0.529083, 0.370239, 0.256844, 0.176983, 0.121304,
      0.082791, 0.056318, 0.038211, 0.025871, 0.017486, 4.188770, 4.188732,
      4.188628, 4.188352, 4.187671, 4.186115, 4.182830, 4.176234, 4.163261,
      4.138284, 4.092055, 4.009206, 3.867171, 3.638731, 3.302580, 2.861651,
      2.355385, 1.846979, 1.392364, 1.019841, 0.731849, 0.517409, 0.361743,
      0.250766, 0.172684, 0.118295, 0.080706, 0.054883, 0.037230, 0.025204,
      0.017035, 4.188759, 4.188702, 4.188551, 4.188171, 4.187294, 4.185421,
      4.181657, 4.174292, 4.160089, 4.133434, 4.084980, 3.998932, 3.852801,
      3.619743, 3.279339, 2.835771, 2.329266, 1.822929, 1.372041, 1.003734,
      0.719559, 0.508296, 0.355130, 0.246037, 0.169346, 0.115966, 0.079096,
      0.053783, 0.036482, 0.024699, 0.016694, 4.188742, 4.188659, 4.188449,
      4.187958, 4.186900, 4.184765, 4.180606, 4.172596, 4.157440, 4.129551,
      4.079183, 3.990499, 3.841031, 3.604270, 3.260519, 2.814871, 2.308170,
      1.803653, 1.355971, 0.991032, 0.709899, 0.501150, 0.349948, 0.242337,
      0.166741, 0.114156, 0.077855, 0.052940, 0.035912, 0.024316, 0.016437,
      4.188719, 4.188603, 4.188330, 4.187736, 4.186534, 4.184200, 4.179711,
      4.171172, 4.155254, 4.126372, 4.074457, 3.983648, 3.831433, 3.591609,
      3.245083, 2.797662, 2.290945, 1.788288, 1.343237, 0.981003, 0.702295,
      0.495533, 0.345879, 0.239437, 0.164709, 0.112756, 0.076902, 0.052295,
      0.035478, 0.024023, 0.016239, 4.188688, 4.188536, 4.188204, 4.187532,
      4.186227, 4.183725, 4.178948, 4.169962, 4.153461, 4.123737, 4.070508,
      3.977891, 3.823388, 3.580958, 3.232035, 2.783281, 2.277011, 1.776032,
      1.333144, 0.973094, 0.696303, 0.491107, 0.342676, 0.237173, 0.163143,
      0.111689, 0.076180, 0.051808, 0.035151, 0.023802, 0.016089};
  const double yb[] = {
      // 100 pts, eps=1e-5, down only
      4.188790,   4.188791,  4.188791,   4.188793,  4.188798,   4.188814,
      4.188859,   4.188986,  4.189328,   4.190189,  4.192213,   4.196874,
      4.208210,   4.236983,  4.307724,   4.473512,  4.852754,   5.723574,
      7.838600,   13.880005, 40.362527,  -1,        -1,         -1,
      -1,         -1,        -1,         -1,        -1,         -1,
      -1,         4.188790,  4.188791,   4.188791,  4.188793,   4.188798,
      4.188813,   4.188858,  4.188982,   4.189319,  4.190165,   4.192156,
      4.196738,   4.207879,  4.236133,   4.305531,  4.467899,   4.838139,
      5.683061,   7.710538,  13.356740,  36.482501, -1,         -1,
      -1,         -1,        -1,         -1,        -1,         -1,
      -1,         -1,        4.188790,   4.188791,  4.188791,   4.188793,
      4.188798,   4.188812,  4.188854,   4.188973,  4.189292,   4.190095,
      4.191984,   4.196332,  4.206887,   4.233594,  4.298997,   4.451212,
      4.795122,   5.566006,  7.352764,   11.994360, 28.085852,  261.834153,
      -1,         -1,        -1,         -1,        -1,         -1,
      -1,         -1,        -1,         4.188790,  4.188791,   4.188791,
      4.188792,   4.188797,  4.188810,   4.188848,  4.188956,   4.189246,
      4.189978,   4.191699,  4.195657,   4.205243,  4.229397,   4.288214,
      4.424093,   4.726265,  5.384525,   6.831926,  10.241985,  19.932261,
      69.581522,  -1,        -1,         -1,        -1,         -1,
      -1,         -1,        -1,         -1,        4.188790,   4.188791,
      4.188791,   4.188792,  4.188796,   4.188807,  4.188840,   4.188933,
      4.189183,   4.189814,  4.191300,   4.194716,  4.202957,   4.223593,
      4.273478,   4.387310,  4.635278,   5.155725,  6.228070,   8.494834,
      13.840039,  30.707995, 217.685066, -1,        -1,         -1,
      -1,         -1,        -1,         -1,        -1,         4.188790,
      4.188791,   4.188791,  4.188792,   4.188795,  4.188803,   4.188829,
      4.188903,   4.189101,  4.189604,   4.190789,  4.193514,   4.200048,
      4.216255,   4.254956,  4.342131,   4.526936,  4.897901,   5.610205,
      6.966144,   9.704274,  16.182908,  37.778925, 387.192994, -1,
      -1,         -1,        -1,         -1,        -1,         -1,
      4.188790,   4.188790,  4.188791,   4.188791,  4.188793,   4.188799,
      4.188817,   4.188866,  4.189002,   4.189348,  4.190168,   4.192054,
      4.196535,   4.207468,  4.233117,   4.289818,  4.406055,   4.627980,
      5.025468,   5.719284,  6.973164,   9.454673,  15.165237,  32.669841,
      175.505441, -1,        -1,         -1,        -1,         -1,
      -1,         4.188790,  4.188790,   4.188790,  4.188791,   4.188791,
      4.188794,   4.188801,  4.188823,   4.188885,  4.189047,   4.189439,
      4.190344,   4.192444,  4.197336,   4.208277,  4.231803,   4.277545,
      4.359611,   4.499274,  4.738728,   5.168746,  5.973516,   7.554064,
      10.974866,  20.015747, 59.729701,  -1,        -1,         -1,
      -1,         -1,        4.188790,   4.188790,  4.188790,   4.188790,
      4.188789,   4.188788,  4.188784,   4.188774,  4.188751,   4.188701,
      4.188602,   4.188390,  4.187803,   4.185972,  4.180908,   4.169592,
      4.145768,   4.102670,  4.041148,   3.980771,  3.961901,   4.039620,
      4.292076,   4.858676,  6.044956,   8.672140,  15.704344,  48.437802,
      -1,         -1,        -1,         4.188790,  4.188790,   4.188790,
      4.188789,   4.188787,  4.188781,   4.188764,  4.188718,   4.188599,
      4.188311,   4.187661,  4.186201,   4.182644,  4.173501,   4.151495,
      4.104610,   4.014334,  3.863341,   3.650600,  3.398722,   3.141206,
      2.906174,   2.713737,  2.580237,   2.524103,  2.574058,   2.785372,
      3.278275,   4.350631,  6.897568,   14.924070, 4.188790,   4.188790,
      4.188790,   4.188788,  4.188784,   4.188773,  4.188742,   4.188656,
      4.188430,   4.187878,  4.186617,   4.183786,  4.177000,   4.160053,
      4.120473,   4.038061,  3.886243,   3.645077,  3.322103,   2.951900,
      2.573084,   2.212134,  1.883264,   1.592189,  1.339766,   1.124235,
      0.942582,   0.791354,  0.667141,   0.566827,  0.487862};
  const double slo = 3, sinc = -0.2;   // logs of values; shi=-3
  const double blor = 0, bincr = 0.2;  // logs of values; bhir=3
  const double blob = 0, bincb = 0.1;  // actual values; bhib=1
  const int snum = 31, bnumr = 16, bnumb = 11;
  double x[4], y[4], z[4];
  int sindx, bindx, i, j;
  double ans;

  if (step < 0 || a < 0) return -1;
  if (step == 0 && b >= 0 && b < 1) return -1;
  if (step == 0) return 0;
  step = log(step / a);
  b /= a;

  sindx = (step - slo) / sinc;
  for (i = 0; i < 4; i++) x[i] = slo + (sindx - 1 + i) * sinc;
  z[0] = (step - x[1]) * (step - x[2]) * (step - x[3]) /
         (-6.0 * sinc * sinc * sinc);
  z[1] = (step - x[0]) * (step - x[2]) * (step - x[3]) /
         (2.0 * sinc * sinc * sinc);
  z[2] = (step - x[0]) * (step - x[1]) * (step - x[3]) /
         (-2.0 * sinc * sinc * sinc);
  z[3] = (step - x[0]) * (step - x[1]) * (step - x[2]) /
         (6.0 * sinc * sinc * sinc);

  if (b < 0)
    for (ans = i = 0; i < 4; i++) {
      if (sindx - 1 + i >= snum)
        ans += z[i] * 2.0 * PI * exp(2.0 * x[i]);
      else if (sindx - 1 + i < 0)
        ans += z[i] * 4.0 * PI / 3.0;
      else
        ans += z[i] * yi[sindx - 1 + i];
    }
  else if (b < 1.0) {
    bindx = (b - blob) / bincb;
    if (bindx < 1)
      bindx = 1;
    else if (bindx + 2 >= bnumb)
      bindx = bnumb - 3;
    while (sindx + 3 >= snum ||
           (sindx > 0 && yb[(bindx - 1) * snum + sindx + 3] < 0))
      sindx--;
    for (i = 0; i < 4; i++) x[i] = slo + (sindx - 1 + i) * sinc;
    z[0] = (step - x[1]) * (step - x[2]) * (step - x[3]) /
           (-6.0 * sinc * sinc * sinc);
    z[1] = (step - x[0]) * (step - x[2]) * (step - x[3]) /
           (2.0 * sinc * sinc * sinc);
    z[2] = (step - x[0]) * (step - x[1]) * (step - x[3]) /
           (-2.0 * sinc * sinc * sinc);
    z[3] = (step - x[0]) * (step - x[1]) * (step - x[2]) /
           (6.0 * sinc * sinc * sinc);
    for (j = 0; j < 4; j++)
      for (y[j] = i = 0; i < 4; i++) {
        if (sindx - 1 + i >= snum)
          y[j] += z[i] * yb[(bindx - 1 + j) * snum];
        else if (sindx - 1 + i < 0)
          y[j] += z[i] * 4.0 * PI / 3.0;
        else
          y[j] += z[i] * yb[(bindx - 1 + j) * snum + sindx - 1 + i];
      }
    for (j = 0; j < 4; j++) x[j] = blob + (bindx - 1 + j) * bincb;
    z[0] =
        (b - x[1]) * (b - x[2]) * (b - x[3]) / (-6.0 * bincb * bincb * bincb);
    z[1] = (b - x[0]) * (b - x[2]) * (b - x[3]) / (2.0 * bincb * bincb * bincb);
    z[2] =
        (b - x[0]) * (b - x[1]) * (b - x[3]) / (-2.0 * bincb * bincb * bincb);
    z[3] = (b - x[0]) * (b - x[1]) * (b - x[2]) / (6.0 * bincb * bincb * bincb);
    ans = z[0] * y[0] + z[1] * y[1] + z[2] * y[2] + z[3] * y[3];
  } else {
    b = log(b);
    bindx = (b - blor) / bincr;
    if (bindx < 1)
      bindx = 1;
    else if (bindx + 2 >= bnumr)
      bindx = bnumr - 3;
    for (j = 0; j < 4; j++)
      for (y[j] = i = 0; i < 4; i++) {
        if (sindx - 1 + i >= snum && b == 0)
          y[j] += z[i] * 2.0 * PI * exp(x[i]) * (1.0 + exp(x[i]));
        else if (sindx - 1 + i >= snum)
          y[j] += z[i] * 2.0 * PI * exp(2.0 * x[i]) * exp(b) / (exp(b) - 1.0);
        else if (sindx - 1 + i < 0)
          y[j] += z[i] * 4.0 * PI / 3.0;
        else
          y[j] += z[i] * yr[(bindx - 1 + j) * snum + sindx - 1 + i];
      }
    for (j = 0; j < 4; j++) x[j] = blor + (bindx - 1 + j) * bincr;
    z[0] =
        (b - x[1]) * (b - x[2]) * (b - x[3]) / (-6.0 * bincr * bincr * bincr);
    z[1] = (b - x[0]) * (b - x[2]) * (b - x[3]) / (2.0 * bincr * bincr * bincr);
    z[2] =
        (b - x[0]) * (b - x[1]) * (b - x[3]) / (-2.0 * bincr * bincr * bincr);
    z[3] = (b - x[0]) * (b - x[1]) * (b - x[2]) / (6.0 * bincr * bincr * bincr);
    ans = z[0] * y[0] + z[1] * y[1] + z[2] * y[2] + z[3] * y[3];
  }
  return ans * a * a * a;
}

double bindingradius(double rate, double dt, double difc, double b, int rel) {
  double a, lo, dif, step;
  int n;

  if (rate < 0 || dt < 0 || difc <= 0) return -1;
  if (rate == 0) return 0;
  if (dt == 0) {
    if (b < 0) return rate / (4 * PI * difc);
    if (rel && b > 1) return rate * (1 - 1 / b) / (4 * PI * difc);
    if (rel && b <= 1) return -1;
    if (b > 0) return rate / (4 * PI * difc + rate / b);
    return -1;
  }
  step = sqrt(2.0 * difc * dt);
  lo = 0;
  a = step;
  while (numrxnrate(step, a, rel ? b * a : b) < rate * dt) {
    lo = a;
    a *= 2.0;
  }
  dif = a - lo;
  for (n = 0; n < 15; n++) {
    dif *= 0.5;
    a = lo + dif;
    if (numrxnrate(step, a, rel ? b * a : b) < rate * dt) lo = a;
  }
  a = lo + 0.5 * dif;
  return a;
}

/* unbindingradius returns the unbinding radius that corresponds to the geminate
reaction probability in pgem, the time step in dt, the mutual diffusion constant
in difc, and the binding radius in a.  Illegal inputs result in a return value
of -2.  If the geminate binding probability can be made as high as that
requested, the corresponding unbinding radius is returned.  Otherwise, the
negative of the maximum achievable pgem value is returned.
Modified 2/25/08 to allow for dt==0.  */
double unbindingradius(double pgem, double dt, double difc, double a) {
  double b, lo, dif, step, ki, kmax;
  int n;

  if (pgem <= 0 || pgem >= 1 || difc <= 0 || a <= 0 || dt < 0) return -2;
  if (dt == 0) return a / pgem;
  step = sqrt(2.0 * difc * dt);
  ki = numrxnrate(step, a, -1);
  kmax = numrxnrate(step, a, 0);
  if (1.0 - ki / kmax < pgem) return ki / kmax - 1.0;
  lo = 0;
  b = a;
  while (1.0 - ki / numrxnrate(step, a, b) > pgem) {
    lo = b;
    b *= 2.0;
  }
  dif = b - lo;
  for (n = 0; n < 15; n++) {
    dif *= 0.5;
    b = lo + dif;
    if (1.0 - ki / numrxnrate(step, a, b) > pgem) lo = b;
  }
  b = lo + 0.5 * dif;
  return b;
}
