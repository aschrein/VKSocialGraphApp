#pragma once
#include "DataStruct.hpp"
#include "ogl.hpp"
#include <qvector.h>
#include <qbitarray.h>
#include <qimage.h>
struct UVMapping
{
	uint8_t u , v , u_size , v_size;
};
struct TextureRequest
{
	int32_t person_id;
	int32_t person_view_id;
	float lod;
};
struct TextureResponse
{
	int32_t person_view_id;
	int32_t texture_id;
	int32_t width , heght;
};
/*static void addRequest( void *uv_requests , TextureRequest request )
{
	( ( TextureRequest* )( ( int32_t* )uv_requests + 1 ) )[ ( ( int32_t* )uv_requests )[ 0 ]++ ] = request;
}
static int32_t getResponsesCount( void *uv_responses )
{
	return ( ( int32_t* )uv_responses )[ 0 ];
}
static TextureResponse getResponse( void *uv_responses , int32_t i )
{
	return ( ( TextureResponse* )( ( int32_t* )uv_responses + 1 ) )[ i ];
}*/
struct Atlas
{
	static constexpr int ATLAS_SIZE = 4096;
	static constexpr int PITCH = ATLAS_SIZE / 8;
	static constexpr int CELL_SIZE = 32;
	static constexpr int CELL_COUNT = ATLAS_SIZE / CELL_SIZE;
	/////
	QVector< UVMapping > mappings;
	//QVector< uint8_t > bitset;
	QVector< uint8_t > bitset;
	uint32_t atlas_texture , fbo , vbo;
	Program program;
	uint32_t utex , uscale , uoffset;
	////
	uint8_t getBit( int32_t j , int32_t i )
	{
		//return (bitset[ ( i * CELL_COUNT + j ) / 8 ] >> ( j - j/ 8 )) & 1;
		return bitset[ i * CELL_COUNT + j ];
	}
	void resetBit( int32_t j , int32_t i )
	{
		//bitset[ ( i * CELL_COUNT + j ) / 8 ] &= ( ~( 1 << ( j - j / 8 ) ) );
		bitset[ i * CELL_COUNT + j ] = 0;
	}
	void setBit( int32_t j , int32_t i )
	{
		//bitset[ ( i * CELL_COUNT + j ) / 8 ] |= ( 1 << ( j - j / 8 ) );
		bitset[ i * CELL_COUNT + j ] = 1;
	}
	void init()
	{
		bitset.resize( CELL_COUNT * CELL_COUNT );
		//memset( &bitset[0] , 0 , CELL_COUNT * CELL_COUNT );
		glGenTextures( 1 , &atlas_texture );
		glBindTexture( GL_TEXTURE_2D , atlas_texture );
		glTexImage2D( GL_TEXTURE_2D , 0 , GL_RGB , ATLAS_SIZE , ATLAS_SIZE , 0 , GL_RGB , GL_UNSIGNED_BYTE , nullptr );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_MAG_FILTER , GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_MIN_FILTER , GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_WRAP_S , GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_WRAP_T , GL_CLAMP_TO_EDGE );
		glGenFramebuffers( 1 , &fbo );
		glBindFramebuffer( GL_FRAMEBUFFER , fbo );
		glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 , GL_TEXTURE_2D , atlas_texture , 0 );
		glViewport( 0 , 0 , ATLAS_SIZE , ATLAS_SIZE );
		glClearColor( 0.5f , 0.5f , 0.5f , 1.0f );
		glClear( GL_COLOR_BUFFER_BIT );
		glBindFramebuffer( GL_FRAMEBUFFER , GL_BACK );
		glBindTexture( GL_TEXTURE_2D , 0 );
		glGenBuffers( 1 , &vbo );
		glBindBuffer( GL_ARRAY_BUFFER , vbo );
		float rect_coords[] =
		{
			-1.0f , -1.0f ,
			-1.0f , 1.0f ,
			1.0f , 1.0f ,
			-1.0f , -1.0f ,
			1.0f , 1.0f ,
			1.0f , -1.0f
		};
		glBufferData( GL_ARRAY_BUFFER , 48 , rect_coords , GL_STATIC_DRAW );
		glBindBuffer( GL_ARRAY_BUFFER , 0 );
		program.init(
			"uniform sampler2D texture;\n\
			varying vec2 uv;\n\
			void main()\n\
			{\n\
				gl_FragColor = vec4( texture2D( texture , uv ).xyz , 1.0 );\n\
			}"
			,
			"uniform vec2 offset;\n\
			uniform vec2 scale;\n\
			attribute vec2 position;\n\
			varying vec2 uv;\n\
			void main()\n\
			{\n\
				uv = vec2( 0.5 , -0.5 ) * position + vec2( 0.5 , 0.5 );\n\
				vec2 toffset = offset * 2.0 - vec2( 1.0 , 1.0 ) + scale;\n\
				gl_Position = vec4( position * scale + toffset , 0.0 , 1.0 );\n\
			}"
		);
		utex = program.getUniform( "texture" );
		uoffset = program.getUniform( "offset" );
		uscale = program.getUniform( "scale" );
	}
	~Atlas()
	{
		glDeleteFramebuffers( 1 , &fbo );
		glDeleteBuffers( 1 , &vbo );
		glDeleteTextures( 1 , &atlas_texture );
	}
	bool test( int32_t x , int32_t y , int32_t width , int32_t height )
	{
		while( true )
		{
			if( x >= CELL_COUNT || y >= CELL_COUNT )
			{
				return false;
			}
			if( width == 1 )
			{
				if( height == 1 )
				{
					return true;
				} else if( !getBit( x , y ) )
				{
					y++;
					height--;
				} else
				{
					return false;
				}
			} else if( !getBit( x , y ) )
			{
				x++;
				width--;
			} else
			{
				return false;
			}
		}
	}
	struct Cell
	{
		uint8_t x , y;
	};
	Cell findNext( int32_t x , int32_t y , uint8_t val )
	{
		while( true )
		{
			if( y >= CELL_COUNT )
			{
				return{ 0, 0 };
			}
			if( x >= CELL_COUNT )
			{
				x = 0;
				y++;
				continue;
			}
			if( getBit( x , y ) == val )
			{
				return{ uint8_t(x),uint8_t(y) };
			} else
			{
				x++;
			}
		}
	}
	Cell findFit( int32_t x , int32_t y , int32_t width , int32_t height )
	{
		Cell cell = { ( uint8_t )x , ( uint8_t )y };
		while( true )
		{
			cell = findNext( cell.x , cell.y , 0 );
			if( cell.x < 0 )
			{
				return cell;
			}
			if( test( cell.x , cell.y , width , height ) )
			{
				return cell;
			}
			cell = findNext( cell.x , cell.y , 1 );
			if( cell.x < 0 )
			{
				return cell;
			}
		}
	}
	/*void print()
	{
		for( int i = 0; i < CELL_COUNT; i++ )
		{
			__android_log_print( ANDROID_LOG_VERBOSE , "NATIVE" , "%i%i%i%i" , ( ( uint32_t* )bitset )[ 4 * i ] , ( ( uint32_t* )bitset )[ 1 + 4 * i ] , ( ( uint32_t* )bitset )[ 2+4 * i ] , ( ( uint32_t* )bitset )[ 3+ 4 * i ] );
		}
	}*/
	int32_t allocate( QImage const &image )
	{
		//auto gl_image = QGLWidget::convertToGLFormat( image );
		if( image.width() == 0 || image.height() == 0 )
		{
			return -1;
		}
		GLuint texture;
		glGenTextures( 1 , &texture );
		glBindTexture( GL_TEXTURE_2D , texture );
		/*glTexSubImage2D( GL_TEXTURE_2D , 0 , 0 , 0 , image.width() ,
			image.height() , GL_RGB , GL_UNSIGNED_BYTE , image.bits() );*/
		glTexImage2D( GL_TEXTURE_2D , 0 , GL_RGBA , image.width() , image.height() , 0 , GL_BGRA , GL_UNSIGNED_BYTE , image.bits() );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_MAG_FILTER , GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_MIN_FILTER , GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_WRAP_S , GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D , GL_TEXTURE_WRAP_T , GL_CLAMP_TO_EDGE );
		auto id = allocate( texture , image.width() , image.height() );
		glDeleteTextures( 1 , &texture );
		return id;
	}
	int32_t allocate( int32_t texture_id , int32_t width , int32_t height )
	{
		int32_t cell_width = ( width + CELL_SIZE / 2 ) / CELL_SIZE;
		int32_t cell_height = ( height + CELL_SIZE / 2 ) / CELL_SIZE;
		Cell cell = findFit( 0 , 0 , cell_width , cell_height );
		if( cell.x < 0 )
		{
			return -1;
		}
		for( int y = 0; y < cell_width; y++ )
		{
			for( int x = 0; x < cell_width; x++ )
			{
				setBit( x + cell.x , y + cell.y );
			}
		}
		mappings.append( { uint8_t( cell.x * 256 / CELL_COUNT ) , uint8_t( cell.y * 256 / CELL_COUNT ) ,
			uint8_t( cell_width * 256 / CELL_COUNT ) , uint8_t( cell_height * 256 / CELL_COUNT ) } );
		program.bind();
		glBindFramebuffer( GL_FRAMEBUFFER , fbo );
		glViewport( 0 , 0 , ATLAS_SIZE , ATLAS_SIZE );
		glBindBuffer( GL_ARRAY_BUFFER , vbo );
		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer( 0 , 2 , GL_FLOAT , false , 8 , 0 );
		glUniform2f( uscale , cell_width / float( CELL_COUNT ) , cell_height / float( CELL_COUNT ) );
		glUniform2f( uoffset , cell.x / float( CELL_COUNT ) , cell.y / float( CELL_COUNT ) );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D , texture_id );
		glUniform1i( utex , 0 );
		glDrawArrays( GL_TRIANGLES , 0 , 6 );
		glDisableVertexAttribArray( 0 );
		
		return mappings.size() - 1;
	}
	void draw()
	{
		program.bind();
		
		//__android_log_print( ANDROID_LOG_VERBOSE , "NATIVE" , "atlas vbo: %i\n" , vbo );
		//__android_log_print( ANDROID_LOG_VERBOSE , "NATIVE" , "atlas texture: %i\n" , atlas_texture );
		glBindBuffer( GL_ARRAY_BUFFER , vbo );
		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer( 0 , 2 , GL_FLOAT , false , 8 , 0 );
		glUniform2f( uscale ,0.3f , 0.3f );
		glUniform2f( uoffset , 0.0f , 0.0f );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D , atlas_texture );
		glUniform1i( utex , 0 );
		glDrawArrays( GL_TRIANGLES , 0 , 6 );
		glDisableVertexAttribArray( 0 );
	}
	void free( int32_t mapping_id )
	{
		auto mapping = mappings[ mapping_id ];
		for( int y = 0; y < mapping.v_size * CELL_COUNT / 256; y++ )
		{
			for( int x = 0; x < mapping.u_size * CELL_COUNT / 256; x++ )
			{
				resetBit( x + mapping.u/4 , y + mapping.v/4 );
			}
		}
	}
};