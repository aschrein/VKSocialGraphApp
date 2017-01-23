#include "VKSocialGraphApp.h"
#include <QtWidgets/QApplication>

//https://oauth.vk.com/authorize?client_id=5784392&display=page&scope=friends&response_type=token&v=5.62
//https://api.vk.com/method/METHOD_NAME?PARAMETERS&access_token=ACCESS_TOKEN
//https://api.vk.com/method/friends.get?count=100&fields=city,photo&access_token=2e0244f56901225b0d45a9ffe2c725a0a5f73e5995362e0f5e0020de3600cd5cc14ccb4ee2a015d9f743b
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	
	VKSocialGraphApp w;
	w.show();
	return a.exec();
}
