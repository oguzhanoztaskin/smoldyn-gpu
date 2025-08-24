/*
 * smolparameters.h
 *
 *  Created on: Oct 20, 2010
 *      Author: denis
 */

#ifndef SMOLPARAMETERS_H_
#define SMOLPARAMETERS_H_

#include	"config.h"

#include <vector_types.h>

#include	<map>

#include	<string>

#include "aabb.h"

#include <ostream>

#include <vector>

#include "smol_cmd.h"

#include "model.h"

namespace	smolgpu
{

class	smolparams_t;

//mol	1	A	-98	-22.5	-9.75
struct	distr_t
{
	int		count;
	float3	pos;
	bool	posRandom[3];

	int		bool3ToInt() const {return (posRandom[0]<<2)+(posRandom[1]<<1)+posRandom[2];}
};

struct	species_t
{
	std::string	name;
	int		id;
	float	difc;
	float3	color;

	std::vector<distr_t>	distr;

	std::string	out_file;

	void	dump(std::ostream&	ostr) const;

	inline int		getNumberOfMols() const
	{
		int count = 0;
		for(size_t i = 0; i < distr.size(); i++)
			count += distr[i].count;

		return count;
	}
};

struct	zeroreact_t
{
	std::string	name;
	std::vector<int>	products;
	float	rate;
	float	prob;

	void	calcProb(float dt, float vol);

	void	dump(std::ostream&	ostr, const smolparams_t&) const;
};

struct	firstreact_t: public	zeroreact_t
{
	int		reactant;
	float	tau;

	void	calcProb(float dt, float vol);
	bool	deallocMolecules() const;
	void	dump(std::ostream&	ostr, const smolparams_t&) const;
};

struct	secondreact_t: public	firstreact_t
{
	int		reactant2;
	float	bindrad2;
	float	unbindrad;

	secondreact_t(): bindrad2(0.0f) {}

	void	dump(std::ostream&	ostr, const smolparams_t&) const;
};

class	smolparams_t
{
public:

	smolparams_t(): accuracy(10), useGraphics(false) {}

	bool	useGraphics;
	int		dim;

	int		max_mol;
	typedef	std::map<std::string,species_t>	species_map_t;
	species_map_t	species;

	int		getSpieceID(const std::string& name) const;
	const	species_t&		getSpiece(int	id) const;
	int		getActualMolsNum() const;
	int		getGridDim() const;

	void	calcReactProb();

	typedef	std::map<std::string, zeroreact_t>	zeroreact_map_t;
	typedef	std::map<std::string, firstreact_t>	firstreact_map_t;
	typedef	std::map<std::string, secondreact_t>	secondreact_map_t;

	typedef	std::map<std::string, Model>	models_map_t;

	typedef	std::vector<SmoldynCmd>	cmds_vector_t;

	cmds_vector_t		cmds;

	zeroreact_map_t		zeroreact;
	firstreact_map_t	firstreact;
	secondreact_map_t	secondreact;
	models_map_t		models;

	void	dump(std::ostream&	ostr) const;

	bool	dumpMolMoments;
	bool	dumpMolCounts;
	bool	dumpMolCountsSpace;

	int		currOrder;

	bbox_t	boundaries;
	float3	csize;
	char	boundCond[3];
	float	startTime;
	float	endTime;
	float	timeStep;

	int		accuracy;

	void	calcParameters();//const std::string& path = "", const std::string& prefix = "");

private:

	void	calcFirstOrderReactProb();
	void	calcSecondOrderReactData();
};

}

#endif /* SMOLPARAMETERS_H_ */
