#pragma once
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <qvector.h>
template< typename T >
struct Array
{
	static constexpr int STEP = 0x20;
	T *data = nullptr;
	int32_t position = 0 , limit = 0;
	void add( T v )
	{
		if( position >= limit )
		{
			auto *new_data = ( T* )malloc( sizeof( T ) * ( limit + STEP ) );
			if( data != nullptr )
			{
				memcpy( new_data , data , sizeof( T ) * limit );
				free( data );
			}
			data = new_data;
			limit += STEP;
		}
		data[ position++ ] = v;
	}
	void dispose()
	{
		if( data )
		{
			position = 0;
			limit = 0;
			free( data );
			data = nullptr;
		}
	}
};
struct QuadNode
{
	int32_t index;
	float x , y , size;
	bool collide( QuadNode const &item )
	{
		return fabsf( this->x - item.x ) < item.size + this->size && fabsf( this->y - item.y ) < item.size + this->size;
	}
};
struct QuadTree
{
	QuadTree *children[ 4 ] = { nullptr };
	QVector< QuadNode > items;
	float x , y , size;
	QuadTree( float x , float y , float size ) :
		x( x ) ,
		y( y ) ,
		size( size )
	{}
	~QuadTree()
	{
		if( children[ 0 ] )
		{
			for( int i = 0; i < 4; i++ )
			{
				delete children[ i ];
			}
		}
	}
	bool collide( QuadNode const &item )
	{
		return fabsf( this->x - item.x ) < item.size + this->size && fabsf( this->y - item.y ) < item.size + this->size;
	}
	void addItem( QuadNode const &item , int MAX_ITEMS = 0x10 , int MAX_DEPTH = 8 )
	{
		if( children[ 0 ] )
		{
			for( int i = 0; i < 4; i++ )
			{
				if( children[ i ]->collide( item ) )
				{
					children[ i ]->addItem( item , MAX_ITEMS , MAX_DEPTH - 1 );
				}
			}
		} else
		{
			items.append( item );
			if( items.size() > MAX_ITEMS && MAX_DEPTH > 0 )
			{
				for( int i = 0; i < 4; i++ )
				{
					float cx = this->x + this->size * ( ( i & 1 ) * 2 - 1 ) * 0.5f;
					float cy = this->y + this->size * ( ( ( i >> 1 ) & 1 ) * 2 - 1 ) * 0.5f;
					children[ i ] = new QuadTree( cx , cy , size / 2 );
					for( int k = 0; k < items.size(); k++ )
					{
						if( children[ i ]->collide( items[ k ] ) )
						{
							children[ i ]->addItem( items[ k ] , MAX_ITEMS , MAX_DEPTH - 1 );
						}
					}
				}
				items.clear();
			}
		}
	}
	void fillColided( float x , float y , float size , QVector< int32_t > &indices )
	{
		if( children[ 0 ] )
		{
			for( int i = 0; i < 4; i++ )
			{
				if( children[ i ]->collide( { 0,x,y,size } ) )
				{
					children[ i ]->fillColided( x , y , size , indices );
				}
			}
		} else
		{
			for( auto &item : items )
			{
				if( item.collide( { 0,x,y,size } ) )
				{
					indices.append( item.index );
				}
			}
		}
	}
	void fillLines( Array< float > &coords )
	{
		if( children[ 0 ] )
		{
			for( int i = 0; i < 4; i++ )
			{
				children[ i ]->fillLines( coords );
			}
		} else
		{
			coords.add( x - size ); coords.add( y - size );
			coords.add( x - size ); coords.add( y + size );
			coords.add( x - size ); coords.add( y + size );
			coords.add( x + size ); coords.add( y + size );
			coords.add( x + size ); coords.add( y + size );
			coords.add( x + size ); coords.add( y - size );
			coords.add( x + size ); coords.add( y - size );
			coords.add( x - size ); coords.add( y - size );
		}
	}
};