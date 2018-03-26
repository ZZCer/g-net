#include<iostream>
#include<string.h>
#include "isolation.h"

using namespace std;

template<class T> isoPtr<T>::isoPtr(int len0)
{
	this->len = len0;
	this->data = (T *)calloc(len0, sizeof(T));
}

//isoPtr++
template<class T>
const isoPtr<T> isoPtr<T>::operator++(int x)
{
	isoPtr<T> tmp = *this;
	
	if (x == 0)
		this->data++;
	else
		this->data += x;

	return tmp;
}

//++isoPtr
template<class T> isoPtr<T>& isoPtr<T>::operator++()
{
	this->data++;
	return *this;
}

//isoPtr_a = isoPtr_b
template<class T> isoPtr<T>& isoPtr<T>::operator=(isoPtr y)
{
	if (this == &y)
		return *this;
	
	free(this->data);
	this->data = (T *)calloc(y.len, sizeof(T));
	memcpy(this->data, y.data, y.len);
	this->len = y.len;
	
	return *this;
}

//reference makes the return value can be left value.
template<class T> T& isoPtr<T>::operator[](int idx)
{
	if (idx >= len) {
		cout<<"index overfolw"<<endl;
		return this->data[0];
	} else {
		return this->data[idx];
	}
}

template<class T> int isoPtr<T>::check_length(int xlen)
{
	if (xlen > this->len)
		return 0;
	
	return 1;
}

/*
int main ()
{
	isoPtr<int> p(10);
	cout<<p.check_length(4)<<endl;
	
	cout<<p[1]<<endl;
	p[1] = 9;
	cout<<p[1]<<endl;

	isoPtr<int> pp(12);
	pp[1] = 11;
	p = pp;
	cout<<p[1]<<endl;

	p++;
	cout<<p[0]<<endl;

	return 0;
}
*/
