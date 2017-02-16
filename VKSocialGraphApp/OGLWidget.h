#pragma once
#include <QWidget>
#include <GL/glew.h>
#include <QOpenGLWidget>
#include <QGLWidget>
#include <View.hpp>
#include <iostream>
#include <QObject>
#include <VKMapping.h>
#include <VKModel.h>
#include <qimage.h>
#include <qmutex.h>
#include <thread>
#include <qeventloop.h>
#include <qtimer.h>
#include <qevent.h>
#include <CudaModule/CudaModule.h>
class OGLWidget : public QOpenGLWidget
{
public:
	OGLWidget( QWidget *parent = 0 ) : QOpenGLWidget( parent )
	{
		QSurfaceFormat fmt;
		fmt.setVersion( 4 , 3 );
		fmt.setSamples( 8 );
		fmt.setProfile( QSurfaceFormat::CompatibilityProfile );
		setFormat( fmt );
		QSurfaceFormat::setDefaultFormat( fmt );
		installEventFilter( this );
	}
	~OGLWidget()
	{
		working = false;
		texture_thread.join();
		packer_thread.join();
	}
	int lastmx = 0 , lastmy = 0;
	bool eventFilter( QObject *obj , QEvent *event )
	{
		if( event->type() == QEvent::MouseButtonPress )
		{
			QMouseEvent *mouseEvent = static_cast<QMouseEvent*>( event );
			lastmx = mouseEvent->pos().x();
			lastmy = mouseEvent->pos().y();
			mx = x - float( lastmx - w / 2 ) / w * z * 2;
			my = ( y - float( lastmy - h / 2 ) / float( h ) * z * 2 ) * h / float( w );
			//spatial_states[ 0 ].x = mx;
			//spatial_states[ 0 ].y = my;
			/*mutex.lock();
			for( uint32_t i = 0; i < model.getPersons().size(); i++ )
			{
				auto person = model.getPersons()[ i ];
				float dx = mx - spatial_states[ i ].x;
				float dy = my - spatial_states[ i ].y;
				if( dx * dx + dy * dy < 4.0 )
				{
					qDebug() << person.first_name << " " << person.last_name << " " << person.vk_id << " " << person.photos_url[ 0 ] << "\n";
				}
			}
			mutex.unlock();*/
		} else if( event->type() == QEvent::MouseMove )
		{
			QMouseEvent *mouseEvent = static_cast<QMouseEvent*>( event );
			
			mx = x - float( lastmx - w / 2 ) / w * z * 2;
			my = ( y - float( lastmy - h / 2 ) / float( h ) * z * 2 ) * h / float( w );
			lastmx = mouseEvent->pos().x();
			lastmy = mouseEvent->pos().y();
			float nmx = x - float( lastmx - w / 2 ) / w * z * 2;
			float nmy = ( y - float( lastmy - h / 2 ) / float( h ) * z * 2 ) * h / float( w );
			if( mouseEvent->buttons() & Qt::LeftButton )
			{
				x -= nmx - mx;
				y -= ( nmy - my ) * w / float( h );
			} else
			{
				z -= nmx - mx;
				z -= ( nmy - my ) * w / float( h );
			}
			mx = nmx;
			my = nmy;
			
			//qDebug() << QString( "Mouse move (%1,%2)\n" ).arg( mouseEvent->pos().x() ).arg( mouseEvent->pos().y() );

		}
		return false;
	}
	Q_OBJECT
protected:
	GLint fbo = -1;
	bool working = true;
	float mx = 0 , my = 0 , x = 0 , y = 0 , z = 100.0f;
	
	std::thread texture_thread , packer_thread;
	void initializeGL()
	{
		packer_thread = std::thread( [ this ]()
		{
			while( working )
			{
				pack();
			}
		} );
		texture_thread = std::thread( [ this ]()
		{
			QEventLoop loop;
			VKMapper vkmapper( "884a977d3a73ae659d7c38475613b3601a2725cff4230de2555f9dca1dc1c43750ffdc9c47a021e25a96c" );
			int requests_counter = 0;
			vkmapper.makeThisThread();
			vkmapper.quitEvent( loop );
			loop.moveToThread( QThread::currentThread() );
			auto timer = new QTimer();
			timer->setInterval( 10 );
			connect( timer , &QTimer::timeout , &loop , &QEventLoop::quit );
			timer->start();
			vkmapper.getUser( [ this , &vkmapper ]( QVector< Person > const &p )
			{
				auto main_user = p[ 0 ];
				mutex.lock();
				addPerson( main_user );
				mutex.unlock();

				vkmapper.getFriends( main_user.vk_id , [ main_user , this , &vkmapper ]( QVector< Person > const &v )
				{
					//ui.widget->findChild< QTreeView* >( QString( "treeView" ) )->setModel( new TreeModel( v ) );
					for( auto const &person : v )
					{
						mutex.lock();
						addPerson( person );
						addRelation( main_user.vk_id , person.vk_id );
						mutex.unlock();
						
						auto id = person.vk_id;
						vkmapper.getFriends( id , [ id , this , &vkmapper ]( QVector< Person > const &v )
						{
							for( auto const &person : v )
							{
								auto id2 = person.vk_id;
								mutex.lock();
								addPerson( person );
								addRelation( id , id2 );
								mutex.unlock();
								
								
								//qDebug() << person.vk_id << " " << fperson.vk_id << "\n";
								vkmapper.getFriends( id2 , [ id2 , this ]( QVector< Person > const &v )
								{
									for( auto const &person : v )
									{
										auto id3 = person.vk_id;
										mutex.lock();
										addPerson( person );
										addRelation( id2 , id3 );
										mutex.unlock();
										
									}
								} );
							}
						} );
					}
					//ui.widget->update();
				} );
			} );
			while( working )
			{
				//uint32_t counter = 0;
				
				for( uint32_t i = 0; i < model.getPersons().size(); i++ )
				{
					//mutex.lock();
					mutex.lock();
					auto person = model.getPersons()[ i ];
					mutex.unlock();
					if( person.uv_mapping_id == -1 )
					{
						if( person.photos_url.size() >= 1 )
						{
							vkmapper.getFile( person.photos_url[ 0 ] , [ person , this , i ]( auto const &bitarr )
							{
								QImage img = QImage::fromData( bitarr , person.photos_url[ 0 ].contains( "png" ) ? "png" : "jpg" );
								if( img.width() == 0 )
								{
									qDebug() << "broken image\n";
									return;
								}
								mutex.lock();
								photos.append( { i , std::move( img ) } );
								mutex.unlock();
								//img.save( person.first_name + ".jpg" );
								//person.uv_mapping_id = atlas.allocate( img );
							} );
							requests_counter++;
						}
						mutex.lock();
						model.getPersons()[ i ].uv_mapping_id = -2;
						mutex.unlock();
					}
				};
				
				//qDebug() << "request made:" << requests_counter << "\n";
				//vkmapper.quitEvent( loop );
				timer->setInterval( 100 );
				connect( timer , &QTimer::timeout , &loop , &QEventLoop::quit );
				timer->start();
				loop.exec();
				//qDebug() << "woked up\n";
				//Sleep( 10 );
			}
		} );
		makeCurrent();
		glewExperimental = true;
		glewInit();
		
		atlas.init();
		view.init();
		glDisable( GL_DEPTH_TEST );
		glEnable( GL_TEXTURE_2D );
		//glDepthFunc ( GL_LEQUAL )
		glDisable( GL_CULL_FACE );
		//glFrontFace ( GL_CW )
		//glCullFace ( GL_BACK )
		glDisable( GL_BLEND );
		glBlendFunc( GL_SRC_ALPHA , GL_ONE_MINUS_SRC_ALPHA );
		glBlendEquation( GL_FUNC_ADD );
	}
	int w , h;
	void resizeGL( int w , int h )
	{
		this->w = w;
		this->h = h;
	}
	QMutex mutex;
	QVector< QPair< uint32_t , QImage > > photos;
	QVector< QPair< uint32_t , UVMapping > > uvmaps;
	void paintGL()
	{
		if( fbo < 0 )
		{
			glGetIntegerv( GL_FRAMEBUFFER_BINDING , &fbo );
		}
		mutex.lock();
		for( auto &photo_pair : photos )
		{
			auto id = atlas.allocate( photo_pair.second );
			if( id < 0 )
			{
				qDebug() << "broken mapping\n";
				continue;
			}
			model.getPersons()[ photo_pair.first ].uv_mapping_id = id;
			uvmaps.append( { photo_pair.first , atlas.mappings[ id ] } );
		}
		photos.clear();
		glBindFramebuffer( GL_FRAMEBUFFER , fbo );
		glViewport( 0 , 0 , w , h );
		glClearColor( 1 , 1 , 1 , 1 );
		glClearDepthf( 1 );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glDisable( GL_DEPTH_TEST );
		glDisable( GL_CULL_FACE );
		glEnable( GL_BLEND );
		glBlendFunc( GL_SRC_ALPHA , GL_ONE_MINUS_SRC_ALPHA );
		glBlendEquation( GL_FUNC_ADD );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D , atlas.atlas_texture );
		//pack();
		view.render( spatial_states , model.getRelations() , aQuadNodes , x , y , z , w , h );
		atlas.draw();
		mutex.unlock();
		/*for( auto const &response : responses )
		{
			auto new_id = atlas.allocate( response.texture_id , response.width , response.heght );
			views.data[ response.person_view_id ].uv_mapping_id = new_id;
		}
		QVector< TextureRequest > requests;

		for( int32_t i = 0; i < views.position; i++ )
		{
			auto person_view = views.data[ i ];
			if( person_view.uv_mapping_id == -1 )
			{
				requests.append( { person_view.person_id , i , 1.0f } );
				views.data[ i ].uv_mapping_id = -2;
			}
		}*/
		
		/*GLuint vbo;
		glGenBuffers( 1 , &vbo );
		glBindBuffer( GL_ARRAY_BUFFER , vbo );
		float rect_coords[] =
		{
			-1.0f , -1.0f , -1.0f , 1.0f , 1.0f , 0.5f
		};
		glBufferData( GL_ARRAY_BUFFER , 24 , rect_coords , GL_STATIC_DRAW );;
		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer( 0 , 2 , GL_FLOAT , GL_FALSE , 8 , 0 );
		glDrawArrays( GL_TRIANGLES , 0 , 3 );
		glDeleteBuffers( 1 , &vbo );*/
		/*glBegin( GL_TRIANGLES );
		glVertex2f( -1.0f , -1.0f );
		glVertex2f( -1.0f , 1.0f );
		glVertex2f( 1.0f , 1.0f );
		glEnd();*/
		update();
	}
private:
	View view;
	PersonGraph model;
	Atlas atlas;
	QVector< SpatialState > spatial_states;
	std::vector< QuadNode > aQuadNodes;
	int32_t spatial_queue = 0;
	void addPerson( Person const &p )
	{
		if( model.addPerson( p ) )
			spatial_queue++;
	}
	void addRelation( uint64_t o0 , uint64_t o1 )
	{
		model.addRelation( o0 , o1 );
	}
	void setUVMapping( uint32_t index , uint8_t u , uint8_t v , uint8_t ru , uint8_t tv )
	{
		//spatial_states[ index ].
	}
	float pushForce( float x )
	{
		return -1.0f / ( 1.0f + x );
	}
	float pullForce( float x )
	{
		return fmin( 1.0f , x );
	}

	void pack()
	{
		
		mutex.lock();
		for( ; spatial_queue > 0; spatial_queue-- )
		{
			float r = sqrtf( unirandf() ) * 100.0f;
			float phi = unirandf() * M_PI * 2;
			SpatialState sstate{ cosf( phi ) * r , sinf( phi ) * r };
			spatial_states.append( sstate );
		}
		for( auto uvmap : uvmaps )
		{
			auto mapping = uvmap.second;
			spatial_states[ uvmap.first ].u = mapping.u;
			spatial_states[ uvmap.first ].u_size = mapping.u_size;
			spatial_states[ uvmap.first ].v = mapping.v;
			spatial_states[ uvmap.first ].v_size = mapping.v_size;
		}
		uvmaps.clear();
		std::vector< vec2 > pos;
		pos.reserve( spatial_states.size() * 2 );
		for( auto sstate : spatial_states )
		{
			pos.push_back( { sstate.x , sstate.y } );
		}
		auto relations = model.getRelations();
		relations.detach();
		mutex.unlock();
		std::vector< QuadNode > aQuadNodes;
		if( pos.size() > 0 )
		{
			packCuda( std::vector< Relation >() , pos , aQuadNodes );
		}
		/*
		
		for( auto const &relation : relations )
		{
			auto &v0 = spatial_states[ relation.first ];
			auto &v1 = spatial_states[ relation.second ];
			//auto &nv0 = nspatial_states[ relation.first ];
			//auto &nv1 = nspatial_states[ relation.second ];
			float dx = v1.x - v0.x;
			float dy = v1.y - v0.y;
			float dist = ( dx * dx + dy * dy );
			if( isfinite( dist ) && fabsf( dist ) > std::numeric_limits< float >::epsilon() )
			{
				dist = sqrtf( dist );
				dx /= dist;
				dy /= dist;
				float force = ( pushForce( dist * 0.05f ) * 2.0f + pullForce( dist ) ) * 1.2f;
				v0.x += dx * force;
				v0.y += dy * force;
				v1.x -= dx * force;
				v1.y -= dy * force;
			}
		}*/
		
		//__android_log_print( ANDROID_LOG_VERBOSE , "NATIVE" , "size(%f,%f,%f)\n" ,
		//	( max_x + min_x ) * 0.5f , ( max_y + min_y ) * 0.5f , fmaxf( fabsf( max_x - min_x ) , fabsf( max_y - min_y ) ) * 0.5f );
		/*QuadTree tree( ( max_x + min_x ) * 0.5f , ( max_y + min_y ) * 0.5f , fmaxf( max_x - min_x , max_y - min_y ) * 0.5f );
		int i = 0;
		for( auto const &sstate : spatial_states )
		{
			tree.addItem( { i++ ,sstate.x,sstate.y,0.1f } );
		}
		QVector< int32_t > indices;
		for( int i = 0; i < spatial_states.size(); i++ )
		{
			auto sstate = spatial_states[ i ];
			auto &nsstate = nspatial_states[ i ];
			indices.clear();
			tree.fillColided( sstate.x , sstate.y , 10.0f , indices );
			//
			for( int32_t j : indices )
			{
				auto sstate1 = spatial_states[ j ];
				auto &nsstate1 = nspatial_states[ j ];
				float dx = sstate1.x - sstate.x;
				float dy = sstate1.y - sstate.y;
				float dist = ( dx * dx + dy * dy );
				if( isfinite( dist ) && fabsf( dist ) > std::numeric_limits< float >::epsilon() && fabsf( dist ) < 100.0f )
				{
					dist = sqrtf( dist );
					dx /= dist;
					dy /= dist;
					float force = pushForce( dist * 2 );
					nsstate.x += dx * force;
					nsstate.y += dy * force;
					nsstate1.x -= dx * force;
					nsstate1.y -= dy * force;
				}
			}
		}*/
		mutex.lock();
		this->aQuadNodes = std::move( aQuadNodes );
		for( int i = 0; i < spatial_states.size(); i++ )
		{
			spatial_states[ i ].x = pos[ i ].x;
			spatial_states[ i ].y = pos[ i ].y;
		}
		mutex.unlock();
		/*float viewproj[] =
		{
		-1.0f , 0.0f , 0.0f , 0.0f ,
		0.0f , float( width ) / height , 0.0f , 0.0f ,
		0.0f , 0.0f , 1.0f , 0.0f ,
		x , -y , 0.0f , z
		};
		Array< float > lines;
		tree.fillLines( lines );
		//__android_log_print( ANDROID_LOG_VERBOSE , "NATIVE" , "lines count %i\n" , lines.position );
		edges.dev_buffer.bind();
		edges.dev_buffer.setData( lines.data , lines.position * sizeof( float ) );
		edges.program.bind();
		glUniform4f( edges.ucolor , 0.0f , 0.0f , 0.0f , 1.0f );
		glUniformMatrix4fv( edges.uviewproj , 1 , false , viewproj );
		glDrawArrays( GL_LINES , 0 , lines.position );
		edges.dev_buffer.unbind();
		lines.dispose();

		*/
	}
};