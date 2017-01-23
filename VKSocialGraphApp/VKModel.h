#pragma once
#include <qstring.h>
#include <qmap.h>
#include <qhash.h>
#include <qset.h>
#include <qvector.h>
struct Person
{
	uint64_t vk_id;
	QString first_name , last_name;
	QVector< QString > photos_url;
};
struct PersonContainer : public Person
{
	int32_t uv_mapping_id;
	PersonContainer( Person const &p ) : Person( p ) , uv_mapping_id( -1 ) {}
	PersonContainer() = default;
	//QSet< uint32_t > friends;
};
inline bool operator==( const PersonContainer &e1 , const PersonContainer &e2 )
{
	return e1.vk_id == e2.vk_id;
}
inline uint qHash( const PersonContainer &key )
{
	return uint( key.vk_id & 0xffffffff );
}
class PersonGraph
{
public:
	bool addPerson( Person const &person )
	{
		if( !id_map.contains( person.vk_id ) )
		{
			persons.append( person );
			id_map[ person.vk_id ] = persons.size() - 1;
			return true;
		}
		return false;
	}
	/*Person getPerson( uint64_t vk_id ) const
	{
		if( id_map.contains( vk_id ) )
		{
			return persons[ id_map[ vk_id ] ];
		} else
		{
			return{ 0 , 0 , "None" , "None" };
		}
	}*/
	void addRelation( uint64_t vk_id0 , uint64_t vk_id1 )
	{
		//persons[ id_map[ vk_id0 ] ].friends.insert( id_map[ vk_id1 ] );
		//persons[ id_map[ vk_id1 ] ].friends.insert( id_map[ vk_id0 ] );
		friends.append( { id_map[ vk_id0 ]  , id_map[ vk_id1 ] } );
	}
	/*auto &removePerson( uint64_t vk_id )
	{
		if( id_map.contains( vk_id ) )
		{
			auto id = id_map[ vk_id ];
			id_map.remove( vk_id );
			auto &person = persons[ id ];
			for( auto fid : person.friends )
			{
				persons[ fid ].friends.remove( id );
			}
			if( id == persons.size() - 1 )
			{
				persons.pop_back();
			} else
			{
				free_cells.append( vk_id );
				persons[ id ] = { 0 , "" , "" };
			}
		}
		return *this;
	}*/
	auto const &getRelations() const
	{
		return friends;
	}
	auto const &getPersons() const
	{
		return persons;
	}
	auto &getPersons()
	{
		return persons;
	}
private:
	QVector< uint32_t > free_cells;
	QHash< uint64_t , uint32_t > id_map;
	QVector< PersonContainer > persons;
	QVector< QPair< uint32_t , uint32_t > > friends;
};