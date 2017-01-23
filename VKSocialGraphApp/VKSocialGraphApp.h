#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_VKSocialGraphApp.h"

class VKSocialGraphApp : public QMainWindow
{
    Q_OBJECT

public:
    VKSocialGraphApp(QWidget *parent = Q_NULLPTR);

private:
    Ui::VKSocialGraphAppClass ui;
};
