#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <cmath>
#include <CudaModule/CudaModule.h>
#include <thrust/device_vector.h>
struct TransientQuadNode
{
	vec2 position;
	float size;
	__host__ __device__ bool contains( vec2 const &point )
	{
		return fabsf( position.x - point.x ) <= size &&  fabsf( position.y - point.y ) <= size;
	}
};
#define cuAssert( x ) if( !(x) ){ return; }
/*
 routine used to map points to nodes and count total occupation per node
*/
__global__ void distributeCells(
	vec2 const *pPoints , int *pPointsToNode , int posN ,
	float centerX , float centerY , float cellSize , QuadNode const *pNodes , int *pCounters )
{
	int pointId = threadIdx.x + blockDim.x * blockIdx.x;

	cuAssert( pointId < posN );

	vec2 point = pPoints[ pointId ];
	QuadNode curNode = pNodes[ 0 ];
	int curIndex = 0 , oldIndex = 0;
	while( curNode.children[ 0 ] > 0 )
	{
		for( int i = 0; i < 4; i++ )
		{
			float childCenterX = centerX + cellSize * ( ( i & 1 ) * 2 - 1 ) / 2;
			float childCenterY = centerY + cellSize * ( ( i >> 1 ) * 2 - 1 ) / 2;
			float childSize = cellSize / 2;
			if(
				TransientQuadNode{ { childCenterX , childCenterY} , childSize }
				.contains( point )
				)
			{
				//no other node contains this point and we must switch to that node and check against its children
				curIndex = curNode.children[ i ];
				curNode = pNodes[ curIndex ];
				centerX = childCenterX;
				centerY = childCenterY;
				cellSize = childSize;
				break;
			}
		}
		cuAssert( curIndex != oldIndex );
		oldIndex = curIndex;
	}
	//increase the node's occupation counter
	atomicAdd( pCounters + curIndex , 1 );
	//map this point to the node
	pPointsToNode[ pointId ] = curIndex;
}
__global__ void kPack( float const *pos , int posN , float *npos )
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i < posN )
	{

	}
}
//device side array
template< class T > using dVector = thrust::device_vector< T >;
//host side array
template< class T > using hVector = std::vector< T >;
/*
allocates 4 more nodes and associate ith node to them as parent
puts indices of the children to "aIndices"
*/
void subdivide(
	dVector< QuadNode > &dQuadNodes ,
	int i ,
	hVector< int > &aLeafsIndices ,
	dVector< int > &dCounters
)
{
	int count = dQuadNodes.size();
	dQuadNodes[ i ] = QuadNode{ count , count + 1 , count + 2 , count + 3 };
	dQuadNodes.push_back( { -1 , -1 , -1 , -1 } );
	dQuadNodes.push_back( { -1 , -1 , -1 , -1 } );
	dQuadNodes.push_back( { -1 , -1 , -1 , -1 } );
	dQuadNodes.push_back( { -1 , -1 , -1 , -1 } );
	aLeafsIndices.push_back( count );
	aLeafsIndices.push_back( count + 1 );
	aLeafsIndices.push_back( count + 2 );
	aLeafsIndices.push_back( count + 3 );
	dCounters.push_back( 0 );
	dCounters.push_back( 0 );
	dCounters.push_back( 0 );
	dCounters.push_back( 0 );
}
void packCuda( hVector< Relation > const &relations ,
	hVector< vec2 > &aPoints ,
	hVector< QuadNode > &out_aQuadNode )
{
	//calculate point system spatial extents
	float max_x = 0.0f , min_x = 0.0f , max_y = 0.0f , min_y = 0.0f;
	for( auto const &pos : aPoints )
	{
		max_x = fmaxf( max_x , pos.x );
		min_x = fminf( min_x , pos.x );
		max_y = fmaxf( max_y , pos.y );
		min_y = fminf( min_y , pos.y );
	}
	float rootX = ( max_x + min_x ) * 0.5f;
	float rootY = ( max_y + min_y ) * 0.5f;
	float rootSize = fmaxf( ( max_x - min_x ) * 0.5f , ( max_y - min_y ) * 0.5f );
	//device side copy of points
	dVector< vec2 > dPoints = aPoints;
	//points to nodes mapping
	dVector< int > dPointsToNodes( aPoints.size() );
	//array of quad nodes
	dVector< QuadNode > dQuadNodes;
	//counter array. one counter per quadnode. used to calculate point occupation and decide whether to divide fat nodes.
	dVector< int > dCounters = hVector< int >{ 0 };
	//indices of leafs in dQuadNodes
	hVector< int > aLeafsIndices;
	//push root node
	dQuadNodes.push_back( { } );
	//allocate 4 leafs
	subdivide( dQuadNodes , 0 , aLeafsIndices , dCounters );
	
	while( true )
	{
		//distribute points among leafs and count leafs occupation
		distributeCells << < dim3( ( aPoints.size() + 31 ) / 32 ) , dim3( 32 , 1 , 1 ) >> > (
			thrust::raw_pointer_cast( dPoints.data() ) ,
			thrust::raw_pointer_cast( dPointsToNodes.data() ) ,
			aPoints.size() ,
			rootX , rootY , rootSize ,
			thrust::raw_pointer_cast( dQuadNodes.data() ) ,
			thrust::raw_pointer_cast( dCounters.data() )
			);
		//split fat leafs. terminate if there are no more fat leafs
		hVector< int > aLeafsIndices_c = aLeafsIndices;
		aLeafsIndices.clear();
		bool split = false;
		for( int leafIndex : aLeafsIndices_c )
		{
			if( dCounters[ leafIndex ] > 10 )
			{
				split = true;
				subdivide( dQuadNodes , leafIndex , aLeafsIndices , dCounters );
			}
		}
		if( !split )
		{
			break;
		}
	}
	out_aQuadNode.resize( dQuadNodes.size() );
	thrust::host_vector< QuadNode > hQuadNodes = dQuadNodes;
	thrust::copy( hQuadNodes.begin() , hQuadNodes.end() , out_aQuadNode.begin() );
	//out_aQuadNode = dQuadNodes;

	//thrust::host_vector< int > counters = dCounters;
	/*thrust::device_vector< vec2 > dnPositions;
	dnPositions.reserve( positions.size() );
	kPack << < dim3( ( posN + 31 ) / 32 ) , dim3( 32 , 1 , 1 ) >> > (
		thrust::raw_pointer_cast< float* >( &dPositions[ 0 ] ) ,
		positions.size() ,
		thrust::raw_pointer_cast< float* >( &dnPositions[ 0 ] )
		);*/
}