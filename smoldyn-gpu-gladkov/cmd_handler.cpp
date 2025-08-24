/*
 * cmd_handler.cpp
 *
 *  Created on: Dec 15, 2010
 *      Author: denis
 */

#include "cmd_handler.h"

#include <fstream>
#include <sstream>

#include "file_path.h"
#include "log_file.h"
#include "smol_solver.h"
#include "smolparameters.h"

using namespace std;

namespace smolgpu {

class molmoments_cmd : public cmd_handler {
 public:
  molmoments_cmd(istringstream& ss, const smolparams_t& p) {
    std::string name = "";
    ss >> name >> filename;

    filename = FilesPath::get().getFilePath(filename);

    specId = p.getSpieceID(name);

    std::ofstream ostr(filename.c_str());
  }

  void dump(std::ostream& out, const smolparams_t& p) const {
    out << "Cmd: molmoments. Dumping " << p.getSpiece(specId).name << " to "
        << filename << "\n";
  }

  void process(SmoldynSolver& sl) {
    std::ofstream ostr(filename.c_str(), std::ios_base::app);

    const std::vector<smolgpu::diff_stat_t>& svec = sl.GetMolMoments();

    for (int i = 0; i < svec.size(); i++) {
      if (svec[i].id == specId) {
        const smolgpu::diff_stat_t& s = svec[i];

        float simTime = sl.GetSimulationTime();

        ostr << simTime << " " << s.count << " " << s.avgPos.x << " "
             << s.avgPos.y << " " << s.avgPos.z << " " << s.m2.x << " "
             << s.m1.x << " " << s.m1.y << " " << s.m1.x << " " << s.m2.y << " "
             << s.m1.z << " " << s.m1.y << " " << s.m1.z << " " << s.m2.z
             << "\n";

        break;
      }
    }
  }

 private:
  int specId;
  std::string filename;
};

class molcount_cmd : public cmd_handler {
 public:
  molcount_cmd(istringstream& ss) {
    ss >> filename;

    filename = FilesPath::get().getFilePath(filename);

    std::ofstream ostr(filename.c_str());
  }

  void dump(std::ostream& out, const smolparams_t& p) const {
    out << "Cmd: molcount. Dumping to " << filename << "\n";
  }

  void process(SmoldynSolver& sl) {
    std::ofstream ostr(filename.c_str(), std::ios_base::app);

    const std::vector<smolgpu::molcount_stat_t>& svec = sl.GetMolCount();

    ostr << sl.GetSimulationTime() << " ";

    for (int i = 0; i < svec.size(); i++) ostr << svec[i].count << " ";

    ostr << "\n";
  }

 private:
  std::string filename;
};
// molcountspace name[(state)] axis low high bins low high low high average
// filename
class molcountspace_cmd : public cmd_handler {
 public:
  molcountspace_cmd(istringstream& ss, const smolparams_t& p) {
    string name;
    ss >> name >> axis;

    if (name != "all")
      specId = p.getSpieceID(name);
    else
      specId = -1;

    switch (axis) {
      case 0:
        ss >> bmin.x >> bmax.x >> bins >> bmin.y >> bmax.y >> bmin.z >> bmax.z;
        break;
      case 1:
        ss >> bmin.y >> bmax.y >> bins >> bmin.x >> bmax.x >> bmin.z >> bmax.z;
        break;
      case 2:
        ss >> bmin.z >> bmax.z >> bins >> bmin.x >> bmax.x >> bmin.y >> bmax.y;
        break;
      default:
        LogFile::get() << "molcountspace cmd: incorrect axis value: " << axis
                       << "\n";
        LogFile::flush();
    }

    ss >> avg >> filename;

    filename = FilesPath::get().getFilePath(filename);

    std::ofstream ostr(filename.c_str());
  }

  void dump(std::ostream& out, const smolparams_t& p) const {
    out << "Cmd: molcountspace. Dumping to " << filename << "\n";
    out << "Axis is: " << axis << " Spieces id: " << specId << " bins: " << bins
        << " averages: " << avg << "\n";
    out << "BMin: " << bmin.x << " " << bmin.y << " " << bmin.z << "\n";
    out << "BMax: " << bmax.x << " " << bmax.y << " " << bmax.z << "\n";
  }

  void process(SmoldynSolver& sl) {
    std::ofstream ostr(filename.c_str(), std::ios_base::app);

    const std::vector<int>& svec =
        sl.GetMolCountSpace(axis, bins, specId, bmin, bmax);

    ostr << sl.GetSimulationTime() << " ";

    for (int i = 0; i < svec.size(); i++) ostr << svec[i] << " ";

    ostr << "\n";
  }

 private:
  std::string filename;
  int specId;
  int axis;  // 0 is x, 1 is y, 2 is z
  int bins;

  float3 bmin;
  float3 bmax;

  bool avg;
};
// equilmol name1[(state1)] name2[(state2)] prob

class equimol_cmd : public cmd_handler {
 public:
  equimol_cmd(istringstream& ss, const smolparams_t& p) {
    std::string name;
    ss >> name;

    spec1 = p.getSpieceID(name);

    ss >> name;

    spec2 = p.getSpieceID(name);

    ss >> prob;
  }

  void process(SmoldynSolver& sl) { sl.ProcessEquimol(spec1, spec2, prob); }

  void dump(std::ostream& out, const smolparams_t& p) const {
    out << "Cmd: equimol\n";
    out << "Molecule 1 Id: " << spec1 << " name: " << p.getSpiece(spec1).name
        << "\n";
    out << "Molecule 2 Id: " << spec2 << " name: " << p.getSpiece(spec2).name
        << "\n";
    out << "Probability: " << prob << "\n";
  }

 private:
  float prob;
  int spec1;
  int spec2;
};

cmd_handler* CreateCmd(istringstream& ss, smolparams_t& p) {
  std::string s;
  ss >> s;

  if (s == "molmoments") {
    p.dumpMolMoments = true;
    return new molmoments_cmd(ss, p);
  }
  if (s == "molcount") {
    p.dumpMolCounts = true;
    return new molcount_cmd(ss);
  }

  if (s == "molcountspace") {
    p.dumpMolCountsSpace = true;
    return new molcountspace_cmd(ss, p);
  }

  if (s == "equilmol") return new equimol_cmd(ss, p);

  LogFile::get() << "Unsupported command: " << s << "\n";
  LogFile::flush();

  return 0;
}

}  // namespace smolgpu
