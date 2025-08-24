/*
 * benchmark.h
 *
 *  Created on: Aug 15, 2010
 *      Author: denis
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#ifdef	_MSC_VER
#define BOOST_USE_WINDOWS_H
#endif

#include <boost/shared_ptr.hpp>

#include <string>

#include <map>

#include <unordered_map>

#include <vector>

#include <stdexcept>

#include <cuda_runtime.h>

#include <ostream>

namespace	cudahelp
{

//Use only for GPU clocking!
class	CudaTimerEvents
{
public:
	~CudaTimerEvents();

	void	start();
	float	stop();
	
	typedef	boost::shared_ptr<CudaTimerEvents> MyType;

	static	MyType	Create() { return MyType(new CudaTimerEvents); }	

private:
	cudaEvent_t startEvent, stopEvent;
	CudaTimerEvents();
};

class	ClockTimer
{
public:
	ClockTimer();
	~ClockTimer();

	void	start();
	float	stop();

	typedef	boost::shared_ptr<ClockTimer> MyType;

	static	MyType	Create() { return MyType(new ClockTimer); }	


private:
	long	time;
};

template	<class	TimingStrategy>
class	Benchmarker
{
public:

	Benchmarker() {}
	~Benchmarker() {}

	void	Start(const std::string& name)
	{
		
		std::pair<timers_map_iterator_t,bool> res = timers_.insert(timers_value_type(name,TimingStrategy::Create()));
		res.first->second->start();
	}

	void	StartAvg(const std::string& name)
	{
		std::pair<timers_map_iterator_t,bool> res = timers_.insert(timers_value_type(name+"@@Avg",TimingStrategy::Create()));
		res.first->second->start();
	}

	void	Stop(const std::string& name)
	{
		timers_map_iterator_t	it = timers_.find(name);
		if(it != timers_.end())
		{
			values_[name] = it->second->stop();
			timers_.erase(it);
		}else
			throw std::invalid_argument("No timer " + name);
	}

	float	GetValue(const std::string& name) const
	{
		values_t::const_iterator cit=values_.find(name);
		if(cit == values_.end())
			throw std::invalid_argument("No value " + name);
		return	cit->second;
	}

	void	StopAvg(const std::string& name)
	{
		timers_map_iterator_t	it = timers_.find(name+"@@Avg");
		if(it != timers_.end())
		{
			float v = it->second->stop();
			averages_[name].push_back(v);
	
			timers_.erase(it);
		}else
			throw std::invalid_argument("No value " + name);
	}

	float	GetAvgValue(const std::string& name) const
	{
		float	avg = 0.0f;

		averages_t::const_iterator	it = averages_.find(name);

		if(it == averages_.end())
			throw std::invalid_argument("No value " + name);

		const std::vector<float>& vals = it->second;

		for(int i = 0; i < vals.size(); i++)
			avg += vals[i];

		if(vals.size() > 1)
			avg /= vals.size();

		return avg;
	}

	void	Print(std::ostream& out) const
	{
		out<<"Benchmark report\nValues:\n";
		{
		values_t::const_iterator	it = values_.begin();
		values_t::const_iterator	end = values_.end();

		while(it != end)
		{
			out<<it->first<<" "<<it->second<<" ms\n";
			it++;
		}
		}
		out<<"Averages:\n";

		averages_t::const_iterator	it = averages_.begin();
		averages_t::const_iterator	end = averages_.end();

		while(it != end)
		{
			try{
				out<<it->first<<" "<<GetAvgValue(it->first)<<" ms\n";
			}catch(std::invalid_argument)
			{
				out<<"Invalid average counter: "<<it->first<<"\n";
			}

			it++;
		}

	}

	void	Reset(bool vals = true, bool avgs = true)
	{
		if(vals)
			values_.clear();

		if(avgs)
			averages_.clear();

		timers_.clear();
	}

private:

	typedef	std::map<std::string,float>	values_t;
	typedef	std::map<std::string,std::vector<float> >	averages_t;
#ifdef	_MSC_VER
	typedef	std::tr1::unordered_map<std::string, typename TimingStrategy::MyType>	timers_map_t;
	typedef typename std::tr1::unordered_map<std::string, typename TimingStrategy::MyType>::iterator	timers_map_iterator_t;
#else
	typedef	std::unordered_map<std::string, typename TimingStrategy::MyType>	timers_map_t;
	typedef	typename std::unordered_map<std::string, typename TimingStrategy::MyType>::iterator	timers_map_iterator_t;	
#endif

	typedef	typename timers_map_t::value_type	timers_value_type;	

	values_t	values_;
	averages_t	averages_;
	timers_map_t	timers_;
};

typedef	Benchmarker<CudaTimerEvents>	CudaBenchmark;
typedef	Benchmarker<ClockTimer>			CpuClockBenchmark;

}

#endif /* BENCHMARK_H_ */
