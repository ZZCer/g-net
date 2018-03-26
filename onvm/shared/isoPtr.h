#ifndef _ISOLATION_H_
#define _ISOLATION_H_

template<class T> class isoPtr {
	T *data;
	int len;

public:
	isoPtr(int len);
	isoPtr& operator++();	//++isoPtr
	const isoPtr operator++(int x);	//isoPtr++
	T& operator[](int idx);
	isoPtr& operator=(isoPtr y);
	int check_length(int xlen);
};

#endif
