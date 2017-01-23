#pragma once
#include <qjsonarray.h>
#include <qjsonobject.h>
#include <qjsondocument.h>
#include <qvector.h>
#include <qobject.h>
#include <functional>
//#include <qnetwork.h>
#include <qnetworkaccessmanager.h>
#include <qnetworkrequest.h>
#include <qmessagebox.h>
#include <qnetworkreply.h>
#include <iostream>
#include <qmessagebox.h>
#include <VKModel.h>
#include <qeventloop.h>
#include <qthread.h>
typedef std::function< void( QVector< Person > const & ) > PersonsCallback;
typedef std::function< void( QByteArray const & ) > FileCallback;
class VKMapper : public QObject
{
	Q_OBJECT
private:
	QString const token;
	QNetworkAccessManager *manager;
	QMap< QNetworkReply* , PersonsCallback > persons_callbacks;
	QMap< QNetworkReply* , FileCallback > file_callbacks;
public:
	VKMapper( QString token ) :
		token( token )
	{
		manager = new QNetworkAccessManager();
		
		connect( manager , &QNetworkAccessManager::finished , this , &VKMapper::handlerReply );
	}
	~VKMapper()
	{
		delete manager;
	}
	void makeThisThread()
	{
		moveToThread( QThread::currentThread() );
		manager->moveToThread( QThread::currentThread() );
	}
	void quitEvent( QEventLoop &loop )
	{
		//connect( manager , &QNetworkAccessManager::finished , &loop , SLOT( quit() ) );
	}
	static bool isValid( QJsonObject const &jsonobj )
	{
		return jsonobj.contains( "first_name" )
			&& jsonobj.contains( "last_name" )
			&& jsonobj.contains( "uid" )
			&& jsonobj.contains( "photo_50" )
			&& jsonobj.contains( "photo_200" )
			&& jsonobj.contains( "photo_max" );
	}
	static Person fromJSONObj( QJsonObject const &jsonobj )
	{
		auto first_name = jsonobj[ "first_name" ].toString();
		auto last_name = jsonobj[ "last_name" ].toString();
		auto vk_id = jsonobj[ "uid" ].toInt();
		auto photo0 = jsonobj[ "photo_50" ].toString();
		auto photo1 = jsonobj[ "photo_200" ].toString();
		auto photo2 = jsonobj[ "photo_max" ].toString();
		return{ uint64_t( vk_id ) , first_name , last_name , { photo0 , photo1 , photo2 } };
	}
	void getUser( PersonsCallback callback )
	{
		QNetworkRequest req( QUrl::fromUserInput( "https://api.vk.com/method/users.get?fields=city,photo_50,photo_200,photo_max&access_token=" + token ) );
		QNetworkReply* reply = manager->get( req );
		
		persons_callbacks[ reply ] = callback;
	}
	void getFile( QString url , FileCallback callback )
	{
		QNetworkRequest req( QUrl::fromUserInput( url ) );
		QNetworkReply* reply = manager->get( req );
		file_callbacks[ reply ] = callback;
	}
	void getFriends( uint64_t vk_id , PersonsCallback callback )
	{
		QNetworkRequest req( QUrl::fromUserInput( "https://api.vk.com/method/friends.get?user_id=" + QString::number( vk_id ) + "&fields=city,photo_50,photo_200,photo_max&access_token=" + token ) );
		QNetworkReply* reply = manager->get( req );
		persons_callbacks[ reply ] = callback;
	}
public slots:
	void handlerReply( QNetworkReply *reply )
	{
		if( persons_callbacks.contains( reply ) )
		{
			auto doc = QJsonDocument::fromJson( reply->readAll() );
			if( doc.toJson().contains( "Too many requests per second" ) )
			{
				QNetworkReply* reply2 = manager->get( QNetworkRequest( reply->request().url() ) );
				persons_callbacks[ reply2 ] = persons_callbacks[ reply ];
				persons_callbacks.remove( reply );
			} else if( !doc.object().contains( "response" ) )
			{
				qDebug() << doc.toJson();
			} else
			{
				auto response = doc.object()[ "response" ].toArray();
				QVector< Person > vec;
				for( auto &obj : response )
				{
					if( !obj.toObject().isEmpty() && isValid( obj.toObject() ) )
					{
						vec.append( fromJSONObj( obj.toObject() ) );
					}
				}
				persons_callbacks[ reply ]( vec );
				persons_callbacks.remove( reply );
			}
		} else if( file_callbacks.contains( reply ) )
		{
			file_callbacks[ reply ]( reply->readAll() );
			file_callbacks.remove( reply );
		}
		reply->deleteLater();
	}
};