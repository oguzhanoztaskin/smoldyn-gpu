/*
 * smolparser.cpp
 *
 *  Created on: Oct 20, 2010
 *      Author: denis
 */

#include "smolparser.h"

#include <string.h>

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "log_file.h"

using namespace std;

namespace smolgpu {

SmoldynParser::SmoldynParser() {}

SmoldynParser::~SmoldynParser() {}

bool parse_line(std::string& fname, istringstream& ss, smolparams_t& params);

void SmoldynParser::operator()(const char* filename, smolparams_t& params) {
  assert(filename);
  ifstream ifs(filename);

  params.dumpMolMoments = false;
  params.dumpMolCounts = false;
  params.dumpMolCountsSpace = false;

  if (!ifs.is_open()) {
    string s = "Can't open file: ";
    s += filename;

    throw std::runtime_error(s);
  }

  char buf[255];

  bool multilineComment = false;

  while (ifs) {
    ifs.getline(buf, 255);

    char* p = strchr(buf, '#');

    char* pp = buf;

    if (p) {
      if (*(p + 1) == '%')
        pp = p + 2;
      else
        *p = 0;
    }

    istringstream ss(pp);
    string cmdname;

    ss >> cmdname;

    if (cmdname == "/*") {
      multilineComment = true;
      continue;
    }

    if (cmdname == "*/") {
      multilineComment = false;
      continue;
    }

    if (cmdname == "end_file") break;

    if (!multilineComment)
      if (cmdname != "" && !parse_line(cmdname, ss, params))
        LogFile::get() << "Unknown token: " << cmdname << "\n";
  }
}

typedef void (*parse_func)(istringstream& ss, smolparams_t& p);

struct parse_entry_t {
  const char* name;
  parse_func func;
};
/*
 * graphics opengl

dim 3
species red green
max_mol 200

difc red 1
difc green 2

color red 1 0 0
color green 0 1 0

time_start 0
time_stop 1000
time_step 0.01

#dim pos1 pos2 type (a,r,p,t)

boundaries 0 0 100 r
boundaries 1 0 100 r
boundaries 2 0 100 r

#nmol name pos0 pos1 ... posdim–1

mol 100 red u u u
mol 100 green u u u

end_file
 *
*class	smolparams_t
{
public:
        bool	useGraphics;
        int		dim;

        int		maxSpecies;
        std::vector<species_t>	species;

        bbox_t	boundaries;
        char	boundCond[3];
        float	startTime;
        float	endTime;
        float	timeStep;
};
 */
//////////////////////////////////////////////////////////////////////////////////////////////////
// FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////////////////

void echo(istringstream& ss, smolparams_t& p) { std::cout << ss.str(); }

void passthru(istringstream& ss, smolparams_t& p) {}

void graphics(istringstream& ss, smolparams_t& p) {
  string type;
  ss >> type;

  p.useGraphics = (type != "none");
}

#define DEFINE_READVAL_FUNC(FuncName, ValueName) \
  void FuncName(istringstream& ss, smolparams_t& p) { ss >> ValueName; }

DEFINE_READVAL_FUNC(dim, p.dim)
DEFINE_READVAL_FUNC(maxSpecies, p.max_mol)
DEFINE_READVAL_FUNC(startTime, p.startTime)
DEFINE_READVAL_FUNC(endTime, p.endTime)
DEFINE_READVAL_FUNC(timeStep, p.timeStep)
DEFINE_READVAL_FUNC(accuracy, p.accuracy)

void boundaries(istringstream& ss, smolparams_t& p) {
  int d = 0;
  ss >> d;
  switch (d) {
    case 0:
      ss >> p.boundaries.min.x >> p.boundaries.max.x >> p.boundCond[0];
      break;
    case 1:
      ss >> p.boundaries.min.y >> p.boundaries.max.y >> p.boundCond[1];
      break;
    case 2:
      ss >> p.boundaries.min.z >> p.boundaries.max.z >> p.boundCond[2];
      break;
    default:
      throw std::runtime_error("Only 0,1,2 are allowed as a dim value");
  }
}

void mol(istringstream& ss, smolparams_t& p) {
  // mol 100 red u u u
  int count = 0;
  string name, xx, yy, zz;
  ss >> count >> name >> xx >> yy >> zz;

  distr_t distr;

  distr.count = count;
  if (xx == "u")
    distr.posRandom[0] = true;
  else {
    distr.posRandom[0] = false;
    distr.pos.x = atof(xx.c_str());
  }

  if (yy == "u")
    distr.posRandom[1] = true;
  else {
    distr.posRandom[1] = false;
    distr.pos.y = atof(yy.c_str());
  }

  if (zz == "u")
    distr.posRandom[2] = true;
  else {
    distr.posRandom[2] = false;
    distr.pos.z = atof(zz.c_str());
  }

  if (name == "all") {
    smolparams_t::species_map_t::iterator it = p.species.begin();
    smolparams_t::species_map_t::iterator end = p.species.end();

    while (it != end) {
      it->second.distr.push_back(distr);

      ++it;
    }
  } else {
    smolparams_t::species_map_t::iterator it = p.species.find(name);
    it->second.distr.push_back(distr);
  }
}

void difc(istringstream& ss, smolparams_t& p) {
  string name;
  float df;
  ss >> name >> df;
  if (name == "all") {
    smolparams_t::species_map_t::iterator it = p.species.begin();
    smolparams_t::species_map_t::iterator end = p.species.end();

    while (it != end) {
      it->second.difc = df;
      ++it;
    }
  } else
    p.species[name].difc = df;
}

void color(istringstream& ss, smolparams_t& p) {
  string name;
  float3 col;
  ss >> name >> col.x >> col.y >> col.z;

  if (name == "all") {
    smolparams_t::species_map_t::iterator it = p.species.begin();
    smolparams_t::species_map_t::iterator end = p.species.end();

    while (it != end) {
      it->second.color = col;
      ++it;
    }
  } else
    p.species[name].color = col;
}

void species(istringstream& ss, smolparams_t& p) {
  string name;
  while (ss >> name) {
    species_t s;
    s.name = name;
    s.id = p.species.size();
    p.species[name] = s;
    float3 c;
    c.x = c.y = c.z = 1.0f;
    p.species[name].color = c;
  }
}

/*
struct	zeroreact_t
{
        std::string	name;
        std::vector<int>	products;
        float	rate;
        float	prob;

        void	dump(std::ostream&	ostr);
};
 */

bool is_number(const std::string& s) {
  for (int i = 0; i < s.size(); i++)
    if ((!isdigit(s[i])) && (s[i] != '.')) return false;
  return true;
}

bool read_products(std::vector<int>& prods, float& rate, istringstream& ss,
                   smolparams_t& p) {
  std::string s = "";

  for (;;) {
    if (!(ss >> s)) break;

    if (s == "+") continue;

    if (s == "0") {
      prods.push_back(-1);
      continue;
    }

    int id = p.getSpieceID(s);

    if (id != -1)
      prods.push_back(id);
    else {
      if (is_number(s))
        rate = atof(s.c_str());
      else {
        LogFile::get() << "Incorrect product: " << s << ". Reaction skipped\n";
        return false;
      }

      return true;
    }
  }

  return true;
}

void load_0th_reaction(const std::string& name, istringstream& ss,
                       smolparams_t& p) {
  zeroreact_t zr;
  zr.name = name;

  if (read_products(zr.products, zr.rate, ss, p)) p.zeroreact[name] = zr;
}

void load_1th_reaction(const std::string& name, const std::string& rct,
                       istringstream& ss, smolparams_t& p) {
  firstreact_t fr;
  fr.name = name;
  int id = p.getSpieceID(rct);

  if (id == -1) {
    LogFile::get() << "Unknown reactant name: " << rct
                   << " skipping reaction\n";
    return;
  }

  fr.reactant = id;

  if (read_products(fr.products, fr.rate, ss, p)) p.firstreact[name] = fr;
}

void load_2nd_reaction(const std::string& name, const std::string& r1,
                       const std::string& r2, istringstream& ss,
                       smolparams_t& p) {
  secondreact_t sr;
  sr.name = name;
  int id1 = p.getSpieceID(r1);

  if (id1 == -1) {
    LogFile::get() << "Unknown reactant name: " << r1 << " skipping reaction\n";
    return;
  }

  sr.reactant = id1;

  int id2 = p.getSpieceID(r2);

  if (id2 == -1) {
    LogFile::get() << "Unknown reactant name: " << r2 << " skipping reaction\n";
    return;
  }

  sr.reactant2 = id2;

  if (read_products(sr.products, sr.rate, ss, p)) p.secondreact[name] = sr;
}

void reaction(istringstream& ss, smolparams_t& p) {
  string name;
  ss >> name;
  string arrow;

  std::string reactants[2];
  int order = 0;

  for (ss >> arrow; arrow != "->"; ss >> arrow)
    if (arrow != "+") {
      reactants[order] = arrow;
      order++;
    }

  if (order == 0) {
    LogFile::get() << "reaction description is wrong. At least 1 reactant is "
                      "required (even 0)\n";
    return;
  }

  if (order == 1) {
    if (reactants[0] == "0")
      load_0th_reaction(name, ss, p);
    else
      load_1th_reaction(name, reactants[0], ss, p);
  } else
    load_2nd_reaction(name, reactants[0], reactants[1], ss, p);
}

void output_files(istringstream& ss, smolparams_t& p) {
  //	string	fname = "";

  //	while(ss>>fname)
  //	{
  //		ofstream	ostr(fname.c_str(), ios_base::out);
  //	}
}

/*
 * cmd e molmoments red diffioutr.txt
cmd e molmoments green diffioutg.txt
cmd e molmoments blue diffioutb.txt
 *
 */

/*
 *
 * low_wall 0 -100 r
        high_wall 0 100 r
        low_wall 1 -42 r
        high_wall 1 42 r
        low_wall 2 -42 r
        high_wall 2 42 r
 */

void process_low_wall(istringstream& ss, smolparams_t& p) {
  int d = 0;
  ss >> d;
  switch (d) {
    case 0:
      ss >> p.boundaries.min.x >> p.boundCond[0];
      break;
    case 1:
      ss >> p.boundaries.min.y >> p.boundCond[1];
      break;
    case 2:
      ss >> p.boundaries.min.z >> p.boundCond[2];
      break;
    default:
      LogFile::get() << "low_wall. Incorrect axis: " << d << "\n";
      LogFile::flush();
  }
}

void process_high_wall(istringstream& ss, smolparams_t& p) {
  int d = 0;
  ss >> d;
  switch (d) {
    case 0:
      ss >> p.boundaries.max.x >> p.boundCond[0];
      break;
    case 1:
      ss >> p.boundaries.max.y >> p.boundCond[1];
      break;
    case 2:
      ss >> p.boundaries.max.z >> p.boundCond[2];
      break;
    default:
      LogFile::get() << "high_wall. Incorrect axis: " << d << "\n";
      LogFile::flush();
  }
}

void process_cmd(istringstream& ss, smolparams_t& p) {
  call_policy* policy = CreateCallPolicy(ss, p);

  p.cmds.push_back(SmoldynCmd(smolgpu::CreateCmd(ss, p), policy));
}

void process_order(istringstream& ss, smolparams_t& p) { ss >> p.currOrder; }

void process_reactant1(int id, istringstream& ss, smolparams_t& p) {
  string rc = "";
  while (ss >> rc) {
    p.firstreact[rc].reactant = id;
    p.firstreact[rc].name = rc;
  }
}

void process_reactant2(int id, int id2, istringstream& ss, smolparams_t& p) {
  string rc = "";
  while (ss >> rc) {
    p.secondreact[rc].reactant = id;
    p.secondreact[rc].reactant2 = id2;

    p.secondreact[rc].name = rc;
  }
}

void process_reactant(istringstream& ss, smolparams_t& p) {
  std::string rc;
  ss >> rc;
  int id = p.getSpieceID(rc);
  int id2 = -1;

  if (p.currOrder == 2) {
    ss >> rc >> rc;
    id2 = p.getSpieceID(rc);

    process_reactant2(id, id2, ss, p);
  } else
    process_reactant1(id, ss, p);
}

void process_rate(istringstream& ss, smolparams_t& p) {
  std::string rc;
  ss >> rc;
  float r;
  ss >> r;

  switch (p.currOrder) {
    case 0:
      p.zeroreact[rc].rate = r;
      p.zeroreact[rc].name = rc;
      break;
    case 1:
      p.firstreact[rc].rate = r;
      p.firstreact[rc].name = rc;
      break;
    case 2:
      p.secondreact[rc].rate = r;
      p.secondreact[rc].name = rc;
      break;
  }
}

void process_product(istringstream& ss, smolparams_t& p) {
  std::string rc;
  ss >> rc;

  std::string prod;

  for (;;) {
    if (!(ss >> prod)) break;

    if (prod == "+") continue;

    int id = p.getSpieceID(prod);

    if (id == -1) break;

    if (prod == "0") id = -1;

    switch (p.currOrder) {
      case 0:
        p.zeroreact[rc].products.push_back(id);
        break;
      case 1:
        p.firstreact[rc].products.push_back(id);
        break;
      case 2:
        p.secondreact[rc].products.push_back(id);
        break;
    }
  }
}

// binding_radius rxn 0.73

void binding_radius(istringstream& ss, smolparams_t& p) {
  string rxn_name = "";
  float rad = 0.0f;

  ss >> rxn_name >> rad;

  p.secondreact[rxn_name].bindrad2 = rad;  //*rad;
}

void mesh(istringstream& ss, smolparams_t& p) {
  string mod, name;
  ss >> mod >> name;
  p.models[mod].LoadRaw(name.c_str(), false);
}

void position(istringstream& ss, smolparams_t& p) {
  string mod;
  vec3_t v;
  ss >> mod >> v.x >> v.y >> v.z;

  p.models[mod].SetPosition(v);
}

void rotation(istringstream& ss, smolparams_t& p) {
  string mod;
  vec3_t v;
  ss >> mod >> v.x >> v.y >> v.z;

  p.models[mod].SetRotation(v);
}

void scale(istringstream& ss, smolparams_t& p) {
  string mod;
  vec3_t v;
  ss >> mod >> v.x >> v.y >> v.z;

  p.models[mod].SetScale(v);
}

void set_type(istringstream& ss, smolparams_t& p) {
  string mod;
  string type;
  ss >> mod >> type;

  uint t = 0;

  if (type == "reflect") t = Reflect;

  if (type == "absorb") t = Absorb;

  if (type == "absorb_prob") {
    t = AbsorbProb;

    float pp = 0;

    ss >> pp;

    p.models[mod].SetProbablility(pp);
  }

  if (type == "transparent") t = Transparent;

  p.models[mod].SetType(t);
}

/*
 * start_reaction
order 1
max_rxn 27

reactant mA ah3
rate ah3 3.4e-2				# mA -> mAp
product ah3 mAp

#% mesh sphere sphere.raw
#% position sphere 0,0,0
#% rotation sphere 0,0,0
#% scale sphere 1,1,1



 */
parse_entry_t g_parse_lookup_table[] = {{"graphics", graphics},
                                        {"dim", dim},
                                        {"max_mol", maxSpecies},
                                        {"time_start", startTime},
                                        {"time_stop", endTime},
                                        {"time_step", timeStep},
                                        {"mol", mol},
                                        {"difc", difc},
                                        {"color", color},
                                        {"species", species},
                                        {"names", species},
                                        {"boundaries", boundaries},
                                        {"reaction", reaction},
                                        {"output_files", output_files},
                                        {"cmd", process_cmd},
                                        {"low_wall", process_low_wall},
                                        {"high_wall", process_high_wall},
                                        {"reactant", process_reactant},
                                        {"rate", process_rate},
                                        {"product", process_product},
                                        {"order", process_order},
                                        {"binding_radius", binding_radius},
                                        {"mesh", mesh},
                                        {"position", position},
                                        {"rotation", rotation},
                                        {"scale", scale},
                                        {"type", set_type},
                                        {"accuracy", accuracy},
                                        {0, 0}};

bool parse_line(std::string& fname, istringstream& ss, smolparams_t& params) {
  for (int i = 0; g_parse_lookup_table[i].name != 0; i++)
    if (fname == g_parse_lookup_table[i].name) {
      g_parse_lookup_table[i].func(ss, params);
      return true;
    }

  return false;
}

}  // namespace smolgpu
