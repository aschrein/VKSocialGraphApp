#pragma once
#include <stdint.h>
#include <GL/glew.h>
#include <qdebug.h>
struct Program
{
	uint32_t program , frag_shader , vert_shader;
	void init( char const *frag_text , char const *vert_text )
	{
		auto compile = []( uint32_t type , char const *raw ) -> int32_t
		{
			uint32_t out = glCreateShader( type );
			glShaderSource( out , 1 , &raw , NULL );
			glCompileShader( out );
			GLint compiled;
			glGetShaderiv( out , GL_COMPILE_STATUS , &compiled );
			if( !compiled )
			{
				GLint length;
				glGetShaderiv( out , GL_INFO_LOG_LENGTH , &length );
				char *log = new char[ length ];
				glGetShaderInfoLog( out , length , &length , log );

				qDebug() << "shader creation error " << log;
				qDebug() << "shader creation error %s\n" << raw;
				glDeleteShader( out );
				delete[] log;
				return 0;
			}
			return out;
		};
		auto frag_shader = compile( GL_FRAGMENT_SHADER , frag_text );
		if( !frag_shader )
		{
			return;
		}
		auto vert_shader = compile( GL_VERTEX_SHADER , vert_text );
		if( !vert_shader )
		{
			glDeleteShader( frag_shader );
			return;
		}
		uint32_t prog = glCreateProgram();
		glAttachShader( prog , frag_shader );
		glAttachShader( prog , vert_shader );
		glLinkProgram( prog );
		GLint compiled;
		glGetProgramiv( prog , GL_LINK_STATUS , &compiled );
		if( !compiled )
		{
			GLint length;
			glGetProgramiv( prog , GL_INFO_LOG_LENGTH , &length );
			char *log = new char[ length ];
			glGetProgramInfoLog( prog , length , &length , &log[ 0 ] );
			qDebug() << "program creation error %s\n" << log;

			glDeleteShader( frag_shader );
			glDeleteShader( vert_shader );
			glDeleteProgram( prog );
			delete[] log;
		};
		this->program = prog;
		this->frag_shader = frag_shader;
		this->vert_shader = vert_shader;
	}
	void bind()
	{
		glUseProgram( program );
	}
	int32_t getUniform( const char *name )
	{
		return glGetUniformLocation( program , name );
	}
	~Program()
	{
		if( program >= 0 )
		{
			glDeleteShader( vert_shader );
			glDeleteShader( frag_shader );
			glDeleteProgram( program );
			vert_shader = 0;
			frag_shader = 0;
			program = 0;
		}
	}
};
struct VertexAttributeGL
{
	uint32_t location;
	uint32_t elem_count;
	uint32_t src_type;
	uint32_t normalized;
	uint32_t stride;
	uint32_t offset;
	uint32_t divisor;
};
struct AttributeArray
{
	uint32_t count;
	VertexAttributeGL attributes[ 10 ];
	void bind()
	{
		for( int i = 0; i < count; i++ )
		{
			glEnableVertexAttribArray( attributes[ i ].location );
			glVertexAttribPointer(
				attributes[ i ].location ,
				attributes[ i ].elem_count ,
				attributes[ i ].src_type ,
				attributes[ i ].normalized ,
				attributes[ i ].stride ,
				( void * )attributes[ i ].offset );
			if( attributes[ i ].divisor )
			{
				glVertexAttribDivisor( attributes[ i ].location , 1 );
			}
		}
	}
	void unbind()
	{
		for( int i = 0; i < count; i++ )
		{
			if( attributes[ i ].divisor )
			{
				glVertexAttribDivisor( attributes[ i ].location , 0 );
			}
			glDisableVertexAttribArray( attributes[ i ].location );
		}
	}
};
struct Buffer
{
	static const int32_t STEP = 10;
	int32_t real_size = 0;
	int32_t size = 0;
	uint32_t bo = 0;
	int32_t target = 0;
	int32_t usage = 0;
	AttributeArray attrib_array = {0};
	int32_t index_type = 0;
	static Buffer createVBO( int usage , AttributeArray attributes )
	{
		Buffer out;
		out.attrib_array = attributes;
		out.target = GL_ARRAY_BUFFER;
		glGenBuffers( 1 , &out.bo );
		out.usage = usage;
		return out;
	}
	static Buffer createIBO( int usage , int index_type )
	{
		Buffer out;
		out.index_type = index_type;
		out.target = GL_ELEMENT_ARRAY_BUFFER;
		glGenBuffers( 1 , &out.bo );
		out.usage = usage;
		return out;
	}
	void resize( int new_size )
	{
		if( new_size > real_size )
		{
			int new_real_size = new_size + STEP;
			if( size > 0 )
			{
				void *tmp = malloc( new_real_size );
				auto map = this->map();
				memcpy( tmp , map , size );
				unmap();
				free( tmp );
				glBufferData( target , new_real_size , tmp , usage );
			} else
			{
				glBufferData( target , new_real_size , NULL , usage );
			}
			real_size = new_real_size;
		} else if( new_size < real_size - STEP )
		{
			void *tmp = malloc( new_size );
			auto map = this->map();
			memcpy( tmp , map , new_size );
			unmap();
			glBufferData( target , new_size , tmp , usage );
			free( tmp );
			real_size = new_size;
		} else if( real_size == 0 )
		{
			glBufferData( target , new_size , NULL , usage );
			real_size = new_size;
		}
		size = new_size;
	}
	template< typename T >
	struct Mapping
	{
		Buffer &buf;
		T *map;
		int loc;
		Mapping( Buffer &b , bool invalidate = false ) : buf( b ) , loc( 0 )
		{
			map = static_cast< T* >( b.map( invalidate ) );
		}
		~Mapping()
		{
			buf.unmap();
		}
		auto &add( T const &a )
		{
			map[ loc++ ] = a;
			return *this;
		}
	};
	void *map( bool invalidate = false )
	{
		void *ptr;
		if( invalidate )
		{
			ptr = glMapBufferRange( target , 0 , size , GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT );
		} else
		{
			ptr = glMapBufferRange( target , 0 , size , GL_MAP_WRITE_BIT | GL_MAP_READ_BIT );
		}
		return ptr;
	}
	void unmap()
	{
		glUnmapBuffer( target );
	}
	void setData( void *data , int size )
	{
		glBufferData( target , size , data , usage );
	}
	void bind()
	{
		glBindBuffer( target , bo );
		attrib_array.bind();
	}
	void bindRaw()
	{
		glBindBuffer( target , bo );
	}
	void unbind()
	{
		glBindBuffer( target , 0 );
		attrib_array.unbind();
	}
	~Buffer()
	{
		if( bo )
		{
			glDeleteBuffers( 1 , &bo );
			bo = 0;
		}
	}
};
template< typename T >
struct Binding
{
	T &b;
	Binding( T &b ) : b( b )
	{
		b.bind();
	}
	~Binding()
	{
		b.unbind();
	}
};