/*
 * log_file.h
 *
 *  Created on: Dec 15, 2010
 *      Author: denis
 */

#ifndef LOG_FILE_H_
#define LOG_FILE_H_

#include <ostream>

namespace	smolgpu
{
	class	LogFile
	{
	public:
		static	void	create(std::ostream&);
		static	LogFile&	get();
		static	void	destroy();
		static	void	flush()
		{
			get().flush_stream();
		}

		template	<class	T>
		LogFile&	operator<<(const T& v)
		{
			out<<v;
			return *this;
		}

		std::ostream&	get_stream() { return out; }
		void	flush_stream() { out.flush(); }

	private:
		static	LogFile*	m_instance;
		std::ostream&	out;
		LogFile(std::ostream& ot):out(ot) {}
		~LogFile() {}
	};
}

#endif /* LOG_FILE_H_ */
