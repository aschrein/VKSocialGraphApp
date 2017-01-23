#pragma once
#include <OGLWidget.h>
#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QTreeView>
#include <qjsonarray.h>
#include <qjsonobject.h>
#include <qjsondocument.h>
#include <VKMapping.h>
class TreeItem
{
public:
	explicit TreeItem( const QList<QVariant> &data , TreeItem *parentItem = 0 )
	{
		parent = parentItem;
		this->items = data;
	}
	~TreeItem()
	{
		qDeleteAll( children );
	}
	void appendChild( TreeItem *child )
	{
		children.append( child );
	}

	TreeItem *child( int row )
	{
		return children.value( row );
	}
	int childCount() const
	{
		return children.count();
	}
	int columnCount() const
	{
		return items.count();
	}
	QVariant data( int column ) const
	{
		return items.value( column );
	}
	int row() const
	{
		if( parent )
			return parent->children.indexOf( const_cast< TreeItem* >( this ) );
		return 0;
	}
	TreeItem *parentItem()
	{
		return parent;
	}

private:
	QList<TreeItem*> children;
	QList<QVariant> items;
	TreeItem *parent;
};
class TreeModel : public QAbstractItemModel
{
	Q_OBJECT

public:
	explicit TreeModel( QVector< Person > const &persons , QObject *parent = 0 ) : QAbstractItemModel( parent )
	{
		QList<QVariant> rootData;
		rootData << "name" << "id";
		rootItem = new TreeItem( rootData );
		for( auto &entry : persons )
		{
			rootItem->appendChild( new TreeItem( QList<QVariant>{ entry.first_name + " " + entry.last_name , entry.vk_id } , rootItem ) );
		}
	}
	~TreeModel()
	{
		delete rootItem;
	}

	QVariant data( const QModelIndex &index , int role ) const Q_DECL_OVERRIDE
	{
		if( !index.isValid() )
			return QVariant();

		if( role != Qt::DisplayRole )
			return QVariant();

		TreeItem *item = static_cast< TreeItem* >( index.internalPointer() );

		return item->data( index.column() );
	}
	Qt::ItemFlags flags( const QModelIndex &index ) const Q_DECL_OVERRIDE
	{
		if( !index.isValid() )
			return 0;
		return QAbstractItemModel::flags( index );
	}
	QVariant headerData( int section , Qt::Orientation orientation ,
		int role = Qt::DisplayRole ) const Q_DECL_OVERRIDE
	{
		if( orientation == Qt::Horizontal && role == Qt::DisplayRole )
			return rootItem->data( section );

		return QVariant();
	}
	QModelIndex index( int row , int column ,
		const QModelIndex &parent = QModelIndex() ) const Q_DECL_OVERRIDE
	{
		if( !hasIndex( row , column , parent ) )
			return QModelIndex();

		TreeItem *parentItem;

		if( !parent.isValid() )
			parentItem = rootItem;
		else
			parentItem = static_cast< TreeItem* >( parent.internalPointer() );

		TreeItem *childItem = parentItem->child( row );
		if( childItem )
			return createIndex( row , column , childItem );
		else
			return QModelIndex();
	}
	QModelIndex parent( const QModelIndex &index ) const Q_DECL_OVERRIDE
	{
		if( !index.isValid() )
			return QModelIndex();

		TreeItem *childItem = static_cast< TreeItem* >( index.internalPointer() );
		TreeItem *parentItem = childItem->parentItem();

		if( parentItem == rootItem )
			return QModelIndex();

		return createIndex( parentItem->row() , 0 , parentItem );
	}
	int rowCount( const QModelIndex &parent = QModelIndex() ) const Q_DECL_OVERRIDE
	{
		TreeItem *parentItem;
		if( parent.column() > 0 )
			return 0;

		if( !parent.isValid() )
			parentItem = rootItem;
		else
			parentItem = static_cast< TreeItem* >( parent.internalPointer() );

		return parentItem->childCount();
	}
	int columnCount( const QModelIndex &parent = QModelIndex() ) const Q_DECL_OVERRIDE
	{
		if( parent.isValid() )
			return static_cast< TreeItem* >( parent.internalPointer() )->columnCount();
		else
			return rootItem->columnCount();
	}

private:
	TreeItem *rootItem;
};