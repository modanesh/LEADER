/*
 * POMDPWorld.cpp
 *
 *  Created on: 20 Sep 2017
 *      Author: panpan
 */

#include <despot/core/pomdp_world.h>

namespace despot {

POMDPWorld::POMDPWorld(DSPOMDP* model, unsigned seed) :
		model_(model),
		random_(Random(seed)) {
}

POMDPWorld::~POMDPWorld() {
}

bool POMDPWorld::Connect() {
	return true;
}

State* POMDPWorld::Initialize() {

	if(FIX_SCENARIO==1)
	{
		std::ifstream fin;fin.open("StartState.txt", std::ios::in);
		state_ = model_->ImportState(fin);
		fin.close();
	}
	else
	{
		state_ = model_->CreateStartState();
	}


	if(FIX_SCENARIO==2)
	{
		std::ofstream fout;fout.open("StartState.txt", std::ios::trunc);
		model_->ExportState(*state_,fout);
		fout.close();
	}

	return state_;
}

despot::State* POMDPWorld::GetCurrentState() {
	return state_;
}

bool POMDPWorld::ExecuteAction(ACT_TYPE action, OBS_TYPE& obs) {
	bool terminal = model_->Step(*state_, random_.NextDouble(), action,
			step_reward_, obs);
	return terminal;
}

} /* namespace despot */
