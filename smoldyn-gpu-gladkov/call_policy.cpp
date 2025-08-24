/*
 * call_policy.cpp
 *
 *  Created on: Dec 17, 2010
 *      Author: denis
 */

#include "call_policy.h"

#include "smolparameters.h"

namespace smolgpu {

class interval_policy : public call_policy {
 public:
  interval_policy(std::istringstream& ss) { ss >> start >> end >> period; }

  bool canCall(float time, int iteration) {
    if (iteration >= start && iteration <= end) return iteration % period == 0;
    return false;
  }

  void dump(std::ostream& out) {
    out << "Interval call policy.\nInterval: " << start << " " << end
        << " Period: " << period << "\n";
  }

 private:
  int start;
  int end;
  int period;
};

class float_interval_policy : public call_policy {
 public:
  float_interval_policy(std::istringstream& ss) : curr(0) {
    ss >> start >> end >> period;
    prev = start;
  }

  bool canCall(float time, int iteration) {
    if (time >= start && time <= end) {
      curr += time - prev;

      prev = time;

      if (curr >= period || time == start) {
        curr = 0;
        return true;
      }
    }
    return false;
  }

  void dump(std::ostream& out) {
    out << "Float interval call policy.\nInterval: " << start << " " << end
        << " Period: " << period << "\n";
  }

 protected:
  float start;
  float end;
  float period;
  float curr;
  float prev;
};

class period_policy : public call_policy {
 public:
  period_policy(std::istringstream& ss) { ss >> period; }

  bool canCall(float time, int iteration) { return iteration % period == 0; }

  void dump(std::ostream& out) {
    out << "Period call policy.\nPeriod: " << period << "\n";
  }

 private:
  int period;
};

class at_time_policy : public call_policy {
 public:
  at_time_policy(std::istringstream& ss) { ss >> tm; }

  bool canCall(float time, int iteration) { return tm == time; }

  void dump(std::ostream& out) {
    out << "At time call policy.\nTime: " << tm << "\n";
  }

 private:
  float tm;
};

class before_policy : public call_policy {
 public:
  before_policy(std::istringstream& ss) {}

  bool canCall(float time, int iteration) { return iteration == 0; }

  void dump(std::ostream& out) { out << "Before call policy\n"; }
};

class after_policy : public call_policy {
 public:
  after_policy(std::istringstream& ss, const smolparams_t& p) {
    endTime = p.endTime - p.timeStep;
  }

  bool canCall(float time, int iteration) { return time >= endTime; }

  void dump(std::ostream& out) { out << "After call policy\n"; }

 private:
  float endTime;
};

// cmd x on off dt xt string

class progression_policy : public float_interval_policy {
 public:
  progression_policy(std::istringstream& ss) : float_interval_policy(ss) {
    ss >> xt;
    p = 0;
    dxt = 0;
  }

  bool canCall(float time, int iteration) {
    if (time >= start && time <= end) {
      curr += time - prev;

      prev = time;

      if (curr + dxt >= period || time == start) {
        curr = 0;

        if (p == 0) dxt = xt;

        if (p > 0) dxt *= xt;

        p++;

        return true;
      }
    }
    return false;
  }

  void dump(std::ostream& out) {
    out << "Progression interval call policy.\nInterval: " << start << " "
        << end << " Period: " << period << " Progression: " << xt << "\n";
  }

 private:
  float xt;
  float dxt;
  int p;
};

call_policy* CreateCallPolicy(std::istringstream& ss, const smolparams_t& p) {
  std::string name;
  ss >> name;

  if (name == "j") return new interval_policy(ss);

  if (name == "x") return new progression_policy(ss);

  if (name == "b") return new before_policy(ss);

  if (name == "a") return new after_policy(ss, p);

  if (name == "i") return new float_interval_policy(ss);

  if (name == "@") return new at_time_policy(ss);

  if (name == "n") return new period_policy(ss);

  if (name != "e") {
    LogFile::get() << "Unsupported call policy: " << name
                   << " command will be executed each iteration\n";
    LogFile::flush();
  }

  return 0;
}

}  // namespace smolgpu
