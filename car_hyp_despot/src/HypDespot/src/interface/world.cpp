/*
 * World.cpp
 *
 *  Created on: 20 Sep 2017
 *      Author: panpan
 */

#include <despot/interface/world.h>

namespace despot {

World::World() {
	state_ = NULL;
}

World::~World() {
}

State* World::GetCurrentState(){
	return NULL;
}

} /* namespace despot */
