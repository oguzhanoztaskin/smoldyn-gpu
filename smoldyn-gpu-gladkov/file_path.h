/*
 * file_path.h
 *
 *  Created on: Dec 16, 2010
 *      Author: denis
 */

#ifndef FILE_PATH_H_
#define FILE_PATH_H_

#include <string>

namespace	smolgpu
{
	class	FilesPath
	{
	public:

		static void	create(const std::string& path = "", const std::string& prefix = "");
		static FilesPath&	get();

		std::string	getFilePath(const std::string& filename);

	private:
		static	FilesPath*	m_instance;

		std::string path;
		std::string prefix;

		FilesPath(const std::string& path, const std::string& prefix);
		~FilesPath();
	};
}

#endif /* FILE_PATH_H_ */
