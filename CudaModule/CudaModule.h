#pragma once
#include <vector>
struct vec2
{
	float x , y;
};
struct QuadNode
{
	/*
	the indices of children within the same array
	or indices of the -1*start( children[ 0 ] ) and of the -1*end( children[ 1 ] ) items if the children[ 0 ] < 0
	*/
	int children[ 4 ];
};
inline vec2 getChildPosition( float centerX , float centerY , float cellSize , int i )
{
	return
	{
		centerX + cellSize * ( ( i & 1 ) * 2 - 1 ) / 2 ,
		centerY + cellSize * ( ( i >> 1 ) * 2 - 1 ) / 2
	};
}
struct Relation
{
	int index0 , index1;
};
extern "C"
{
	__declspec( dllexport ) void packCuda(
		std::vector< Relation > const &aRelations ,
		std::vector< vec2 > &inout_aPoints ,
		std::vector< QuadNode > &out_aQuadNode
	);
}