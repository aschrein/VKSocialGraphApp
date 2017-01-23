#pragma once
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "DataStruct.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include "Atlas.hpp"
#include <qvector.h>
struct SpatialState
{
	float x , y;
	uint8_t u , v , u_size , v_size;
};
struct RectData
{
	float x , y , scale;
	uint8_t u , v , u_size , v_size;
};
static inline float unirandf()
{
	return float( rand() ) / RAND_MAX;
}
struct View
{
	struct Rects
	{
		Program program;
		Buffer dev_buffer , quad_buffer;
		uint32_t vao , uviewproj , utexture , ucolor;
		void init()
		{
			glGenVertexArrays( 1 , &vao );
			glBindVertexArray( vao );
			dev_buffer = Buffer::createVBO( GL_STREAM_DRAW ,
			{ 3 ,
			{
				{ 1 , 2 , GL_FLOAT , GL_FALSE , 16 , 0 , 1 },
				{ 2 , 1 , GL_FLOAT , GL_FALSE , 16 , 8 , 1 },
				{ 3 , 4 , GL_UNSIGNED_BYTE , GL_TRUE , 16 , 12 , 1 }
			}
			} );
			dev_buffer.bind();
			quad_buffer = Buffer::createVBO( GL_STATIC_DRAW ,
			{ 1 ,
			{
				{ 0 , 2 , GL_FLOAT , GL_FALSE , 8 , 0 , 0 },
			}
			} );
			quad_buffer.bind();
			float rect_coords[] =
			{
				-1.0f , -1.0f ,
				-1.0f , 1.0f ,
				1.0f , 1.0f ,
				-1.0f , -1.0f ,
				1.0f , 1.0f ,
				1.0f , -1.0f
			};
			quad_buffer.setData( rect_coords , sizeof( rect_coords ) );
			glBindVertexArray( 0 );
			program.init(
				"#version 430 core\n\
				uniform vec4 color;\n\
				uniform sampler2D texture;\n\
				in vec4 uv;\n\
				void main()\n\
				{\n\
					float alpha = length( uv.zw - 0.5 ) < 0.5 ? 1.0 : 0.0;\n\
					gl_FragColor = color * texture2D( texture , uv.xy ) * vec4( 1.0 , 1.0 , 1.0 , alpha );\n\
				}"
				,
				"#version 430 core\n\
				uniform mat4 viewproj;\n\
				layout (location = 0) in vec2 position;\n\
				layout (location = 1) in vec2 offset;\n\
				layout (location = 2) in float scale;\n\
				layout (location = 3) in vec4 vertex_uv;\n\
				flat out int InstanceID;\n\
				out vec4 uv;\n\
				void main()\n\
				{\n\
					uv.xy = ( position * 0.5 + 0.5 ) * vertex_uv.zw + vertex_uv.xy;\n\
					uv.zw = position * 0.5 + 0.5;\n\
					InstanceID = gl_InstanceID;\n\
					gl_Position = viewproj * vec4( position * scale + offset , 0.0 , 1.0 );\n\
				}"
			);
			uviewproj = program.getUniform( "viewproj" );
			ucolor = program.getUniform( "color" );
			utexture = program.getUniform( "texture" );
		}
		void update( QVector< SpatialState > const &sstates )
		{
			float view_size = 1.1f;
			dev_buffer.resize( sstates.size() * sizeof( RectData ) );
			Buffer::Mapping mapping( dev_buffer , true );
			RectData *rects = ( RectData* )mapping.map;
			int i = 0;
			for( auto const &sstate : sstates )
			{
				RectData rect =
				{
					sstate.x , sstate.y , view_size , sstate.u , sstate.v , sstate.u_size , sstate.v_size
				};
				rects[ i++ ] = rect;
			}
		}
		void draw( QVector< SpatialState > const &sstates , float *viewproj )
		{
			program.bind();
			glUniform4f( ucolor , 1.0f , 1.0f , 1.0f , 1.0f );
			glUniform1i( utexture , 0 );
			glUniformMatrix4fv( uviewproj , 1 , false , viewproj );
			dev_buffer.bindRaw();
			update( sstates );
			glBindVertexArray( vao );
			glDrawArraysInstanced( GL_TRIANGLES , 0 , 6 , sstates.size() );
			glBindVertexArray( 0 );
		}
	} rects;
	struct
	{
		Program program;
		Buffer dev_buffer;
		uint32_t uviewproj , ucolor;
		void init()
		{
			dev_buffer = Buffer::createVBO( GL_STREAM_DRAW , { 1 ,{ { 0 , 2 , GL_FLOAT , false , 8,0 } } } );
			program.init(
				"precision highp float;\n\
				uniform vec4 color;\n\
				void main()\n\
				{\n\
					gl_FragColor = color;\n\
				}"
				,
				"uniform mat4 viewproj;\n\
				attribute vec2 position;\n\
				void main()\n\
				{\n\
					gl_Position = viewproj * vec4( position , 0.0 , 1.0 );\n\
				}"
			);
			uviewproj = program.getUniform( "viewproj" );
			ucolor = program.getUniform( "color" );
		}
		void update( QVector< SpatialState > const &sstates ,
			QVector< QPair< uint32_t , uint32_t > > const &relations )
		{
			dev_buffer.resize( relations.size() * 16 );
			float *lines = ( float* )dev_buffer.map( true );
			int i = 0;
			//__android_log_print( ANDROID_LOG_VERBOSE , "NATIVE" , "edges buf pointer %i\n" , out_edges_buf );
			for( auto const &relation : relations )
			{
				if( relation.first >= sstates.size() || relation.second >= sstates.size() )
				{
					continue;
				}
				auto v0 = sstates[ relation.first ];
				auto v1 = sstates[ relation.second ];
				lines[ i++ ] = v0.x;
				lines[ i++ ] = v0.y;
				lines[ i++ ] = v1.x;
				lines[ i++ ] = v1.y;
			}
			dev_buffer.unmap();
		}
		void draw( QVector< SpatialState > const &sstates ,
			QVector< QPair< uint32_t , uint32_t > > const &relations , float *viewproj )
		{
			if( relations.size() == 0 )
			{
				return;
			}
			dev_buffer.bind();
			update( sstates , relations );
			program.bind();
			glUniform4f( ucolor , 0.0f , 0.0f , 0.0f , 1.0f );
			glUniformMatrix4fv( uviewproj , 1 , false , viewproj );
			glDrawArrays( GL_LINE_STIPPLE , 0 , relations.size() * 2 );
			dev_buffer.unbind();
		}
	} edges;
	void init()
	{
		rects.init();
		edges.init();
	}
	View() = default;
	void render( QVector< SpatialState > const &sstates ,
		QVector< QPair< uint32_t , uint32_t > > const &relations ,
		float x , float y , float z , int width , int height )
	{
		if( sstates.size() == 0 )
		{
			return;
		}
		float viewproj[] =
		{
			-1.0f , 0.0f , 0.0f , 0.0f ,
			0.0f , float( width ) / height , 0.0f , 0.0f ,
			0.0f , 0.0f , 1.0f , 0.0f ,
			x , -y , 0.0f , z
		};
		//edges.draw( sstates , relations , viewproj );
		rects.draw( sstates , viewproj );

	}
};