/*
 * file_path.cpp
 *
 *  Created on: Dec 16, 2010
 *      Author: denis
 */

#include "file_path.h"

#include <stdexcept>

namespace smolgpu
{

FilesPath*	FilesPath::m_instance = 0;

void	FilesPath::create(const std::string& path, const std::string& prefix)
{
	if(m_instance)
		delete m_instance;

	m_instance = new FilesPath(path, prefix);
}


FilesPath&	FilesPath::get()
{
	if(m_instance)
		return *m_instance;
	else
		throw	std::runtime_error("FilesPath is not initialized");

}

std::string	FilesPath::getFilePath(const std::string& filename)
{
	std::string new_fname = filename;

	if(prefix != "")
	{
		std::string	fname = filename;
		size_t s = fname.find_last_of(".");
		std::string ext = fname.substr(s+1);
		std::string file = fname.substr(0,s);

		new_fname = file + "_"+prefix+"."+ext;
	}

	return path + new_fname;
}

FilesPath::FilesPath(const std::string& ph, const std::string& pf): path(ph), prefix(pf)
{

}

FilesPath::~FilesPath()
{

}

}
