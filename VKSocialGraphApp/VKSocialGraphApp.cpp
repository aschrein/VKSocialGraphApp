#include "VKSocialGraphApp.h"

VKSocialGraphApp::VKSocialGraphApp(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	/*QFile file( "test.json" );
	file.open( QIODevice::ReadOnly );
	auto bytes = file.readAll();
	auto doc = QJsonDocument::fromJson( bytes ).object();
	for( auto &person : v )
	{
	qDebug() << person.first_name << " ";
	}
	file.close();*/


	//QtWebView::initialize();
	/*webview = new QWebEngineView( this );
	//set position and size
	webview->setGeometry( 0 , 0 , 200 , 200 );
	webview->load( QUrl( "https://oauth.vk.com/authorize?client_id=5784392&display=page&scope=friends&response_type=token&v=5.62" ) );
	QFrame *frame = new QFrame();
	auto layout = new QBoxLayout( QBoxLayout::Direction::Down , frame );


	layout->addWidget( webview );
	layout->activate();
	frame->show();*/
	//ui.verticalLayout->addWidget( webview );
}
