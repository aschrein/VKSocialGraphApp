#include "VKSocialGraphApp.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    VKSocialGraphApp w;
    w.show();
    return a.exec();
}
