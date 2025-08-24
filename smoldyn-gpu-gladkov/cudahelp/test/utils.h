#ifndef	UTILS_H
#define UTILS_H

#include <string>

#include <stdlib.h>

inline int		int_rand()
{
	return rand()%10000 + 1;
}

void	generate_random(int*	data, int size);

inline	std::string	bool2str(bool b)
{
	return (b)?"yes":"no";
}

#endif
