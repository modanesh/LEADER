#include <despot/core/pomdp_world.h>
#include <despot/plannerbase.h>
#include <despot/solver/baseline_solver.h>
#include <despot/util/seeds.h>

using namespace std;

namespace despot {

void DisableBufferedIO(void) {
	setbuf(stdout, NULL);
	setbuf(stdin, NULL);
	setbuf(stderr, NULL);
	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stdin, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);
}

option::Descriptor* BuildUsage(string lower_bounds_str,
		string base_lower_bounds_str, string upper_bounds_str,
		string base_upper_bounds_str) {
	static string lb_str =
			"-l <arg>  \t--lbtype <arg>  \tLower bound strategy ("
					+ lower_bounds_str + ").";
	static string blb_str = "  \t--blbtype <arg>  \tBase lower bound ("
			+ base_lower_bounds_str + ").";
	static string ub_str =
			"-u <arg>  \t--ubtype <arg>  \tUpper bound strategy ("
					+ upper_bounds_str + ").";
	static string bub_str = "  \t--bubtype <arg>  \tBase upper bound ("
			+ base_upper_bounds_str + ").";
	// option::Arg::Required is a misnomer. The program won't complain if these
	// are absent, and required flags must be checked manually.
	static option::Descriptor usage[] =
			{ { E_HELP, 0, "", "help", option::Arg::None,
					"  \t--help\tPrint usage and exit." },
					{ E_PARAMS_FILE, 0, "m", "model-params",
							option::Arg::Required,
							"-m <arg>  \t--model-params <arg>  \tPath to model-parameters file, if "
									"any." },
					{ E_SIZE, 0, "", "size", option::Arg::Required,
							"  \t--size <arg>  \tSize of a problem (problem specific)." },
					{ E_NUMBER, 0, "", "number", option::Arg::Required,
							"  \t--number <arg>  \tNumber of elements of a problem (problem "
									"specific)." },
					{ E_DEPTH, 0, "d", "depth", option::Arg::Required,
							"-d <arg>  \t--depth <arg>  \tMaximum depth of search tree (default 90)." },
					{ E_DISCOUNT, 0, "g", "discount", option::Arg::Required,
							"-g <arg>  \t--discount <arg>  \tDiscount factor (default 0.95)." },
					{ E_TIMEOUT, 0, "t", "timeout", option::Arg::Required,
							"-t <arg>  \t--timeout <arg>  \tSearch time per move, in seconds (default "
									"1)." },
					{ E_NUMPARTICLES, 0, "n", "nparticles",
							option::Arg::Required,
							"-n <arg>  \t--nparticles <arg>  \tNumber of particles (default 500)." },
					{ E_PRUNE, 0, "p", "prune", option::Arg::Required,
							"-p <arg>  \t--prune <arg>  \tPruning constant (default no pruning)." },
					{ E_GAP, 0, "", "xi", option::Arg::Required,
							"  \t--xi <arg>  \tGap constant (default to 0.95)." },
					{ E_MAX_POLICY_SIM_LEN, 0, "", "max-policy-simlen",
							option::Arg::Required,
							"  \t--max-policy-simlen <arg>  \tDepth to simulate the default policy "
									"until. (default 90)." },
					{ E_SEED, 0, "r", "seed", option::Arg::Required,
							"-r <arg>  \t--seed <arg>  \tRandom number seed (default is random)." },
					{ E_SIM_LEN, 0, "s", "simlen", option::Arg::Required,
							"-s <arg>  \t--simlen <arg>  \tNumber of steps to simulate. (default 90; 0 "
									"= infinite)." },
					{ E_RUNS, 0, "", "runs", option::Arg::Required,
							"  \t--runs <arg>  \tNumber of runs. (default 1)." },
					{ E_LBTYPE, 0, "l", "lbtype", option::Arg::Required,  lb_str.c_str() }, 
					{ E_BLBTYPE, 0, "", "blbtype", option::Arg::Required, blb_str.c_str() }, 
					{ E_UBTYPE, 0, "u", "ubtype", option::Arg::Required,  ub_str.c_str() }, 
					{ E_BUBTYPE, 0, "", "bubtype", option::Arg::Required, bub_str.c_str() },
					{ E_BELIEF, 0, "b", "belief", option::Arg::Required,
							"-b <arg>  \t--belief <arg>  \tBelief update strategy, if applicable." },
					{ E_NOISE, 0, "", "noise", option::Arg::Required,
							"  \t--noise <arg>  \tNoise level for transition in POMDPX belief update." },
					{ E_VERBOSITY, 0, "v", "verbosity", option::Arg::Required,
							"-v <arg>  \t--verbosity <arg>  \tVerbosity level." },
					{ E_SILENCE, 0, "", "silence", option::Arg::None,
							"  \t--silence  \tReduce default output to minimal." },
					{ E_SOLVER, 0, "", "solver", option::Arg::Required,
							"  \t--solver <arg>  \t" }, 
					{ E_PRIOR, 0, "",	"prior", option::Arg::Required,
							"  \t--prior <arg>  \tPOMCP prior." }, 
					{ E_GPU, 0, "", "GPU", option::Arg::Required,
					    "  \t--GPU <arg>  \tEnable GPU support (default false)." },
					{ E_GPUID, 0, "", "GPUID", option::Arg::Required,
						"  \t--GPUID <arg>  \tChoose GPU to use (default 0)." },
					{ E_CPU, 0, "", "CPU", option::Arg::Required,
						"  \t--CPU <arg>  \tEnable CPU multi_thread support (default false)." },
					{ E_THREAD_NUM, 0, "", "num_thread", option::Arg::Required,
						"  \t--num_thread <arg>  \tnum of CPU threads to use (default 0)." },
					{ E_EXPLORATION, 0, "", "explore", option::Arg::Required,
						"  \t--explore <arg>  \tExploration scheme (Vloss or UCT) (default Vloss)." },
					{ E_EXPLORE_CONST, 0, "", "econst", option::Arg::Required,
						"  \t--econst <arg>  \tExploration constant for action branches (default 0.95)." },
					{ E_EXPLORE_CONST_O, 0, "", "oeconst", option::Arg::Required,
						"  \t--oeconst <arg>  \tExploration constant for observation branches(default 0.05)." },
					{ E_DESPOT_THREAD, 0, "", "tdes", option::Arg::Required,
						"  \t--tdes <arg>  \tEnable periodical despot thread (default false)." },
					{ E_EXP_MODE, 0, "", "exp", option::Arg::Required,
						"  \t--exp <arg>  \tUse experiment mode (default false)." },
					{ E_THREAD_GAP, 0, "", "freq", option::Arg::Required,
						"  \t--freq <arg>  \tFrequency launching a despot thread (default large number)." },
					{ E_SWITCH_THRESH, 0, "", "switch", option::Arg::Required,
						"  \t--switch <arg>  \tThreshold of num particles to switch to CPU expansions (default 2)." },
					{ 0, 0, 0, 0, 0, 0 }
			};
	return usage;
}

PlannerBase::PlannerBase(string lower_bounds_str, string base_lower_bounds_str,
		string upper_bounds_str, string base_upper_bounds_str) {
	usage = BuildUsage(lower_bounds_str, base_lower_bounds_str,
			upper_bounds_str, base_upper_bounds_str);
}

PlannerBase::~PlannerBase() {
}

World* PlannerBase::InitializePOMDPWorld(string& world_type, DSPOMDP *model,
		option::Option* options) {
	//Create world: use POMDP model
	world_type = "pomdp";
	POMDPWorld* world = new POMDPWorld(model, Seeds::Next());
	//Establish connection: do nothing
	world->Connect();
	//Initialize: create start state
	world->Initialize();
	return world;
}

option::Option* PlannerBase::InitializeParamers(int argc, char *argv[],
		std::string& solver_type, bool& search_solver, int& num_runs,
		std::string& world_type, std::string& belief_type, int& time_limit) {
	EvalLog::curr_inst_start_time = get_time_second();

	const char *program = (argc > 0) ? argv[0] : "despot";

	argc -= (argc > 0);
	argv += (argc > 0); // skip program name argv[0] if present

	option::Stats stats(usage, argc, argv);
	option::Option *options = new option::Option[stats.options_max];
	option::Option *buffer = new option::Option[stats.buffer_max];
	option::Parser parse(usage, argc, argv, options, buffer);

	/* =========================================
	 * Problem specific default parameter values
	 *=========================================*/
	InitializeDefaultParameters();

	/* =========================
	 * Parse optional parameters
	 * =========================*/
	if (options[E_HELP]) {
		cout << "Usage: " << program << " [options]" << endl;
		option::printUsage(std::cout, usage);
		return NULL;
	}
	OptionParse(options, num_runs, world_type, belief_type, time_limit,
			solver_type, search_solver);

	/* =========================
	 * Global random generator
	 * =========================*/
	Seeds::root_seed(Globals::config.root_seed);
	//unsigned world_seed = Seeds::Next();
	unsigned seed = Seeds::Next();
	Random::RANDOM = Random(seed);

	return options;
}

Solver *PlannerBase::InitializeSolver(DSPOMDP *model, Belief* belief,
		string solver_type, option::Option *options) {
	Solver *solver = NULL;
	// DESPOT or its default policy
	if (solver_type == "DESPOT" || solver_type == "PLB") // PLB: particle lower bound
			{
		string blbtype =
				options[E_BLBTYPE] ? options[E_BLBTYPE].arg : "DEFAULT";
		string lbtype = options[E_LBTYPE] ? options[E_LBTYPE].arg : "DEFAULT";
		ScenarioLowerBound *lower_bound = model->CreateScenarioLowerBound(
				lbtype, blbtype);

		try{
			logi << "Created lower bound " << typeid(*lower_bound).name() << endl;
		}catch (std::exception e) {
			cerr << e.what() << endl;
		}

		logi << "after typeid"<< endl;

		if (solver_type == "DESPOT") {

			string bubtype, ubtype;
			try{
				bubtype =
						options[E_BUBTYPE] ? options[E_BUBTYPE].arg : "DEFAULT";
				ubtype =
						options[E_UBTYPE] ? options[E_UBTYPE].arg : "DEFAULT";
			}
			catch (std::exception e) {
				cerr << e.what() << endl;
			}

			logi << "before CreateScenarioUpperBound" << endl;
			ScenarioUpperBound *upper_bound = model->CreateScenarioUpperBound(
					ubtype, bubtype);

			logi << "Created upper bound " << typeid(*upper_bound).name()
					<< endl;

			solver = new DESPOT(model, lower_bound, upper_bound);
		} else
			solver = new ScenarioBaselineSolver(lower_bound);
	} // AEMS or its default policy
	else if (solver_type == "AEMS" || solver_type == "BLB") {
		string lbtype = options[E_LBTYPE] ? options[E_LBTYPE].arg : "DEFAULT";
		BeliefLowerBound *lower_bound =
				static_cast<BeliefMDP *>(model)->CreateBeliefLowerBound(lbtype);

		logi << "Created lower bound " << typeid(*lower_bound).name() << endl;

		if (solver_type == "AEMS") {
			string ubtype =
					options[E_UBTYPE] ? options[E_UBTYPE].arg : "DEFAULT";
			BeliefUpperBound *upper_bound =
					static_cast<BeliefMDP *>(model)->CreateBeliefUpperBound(
							ubtype);

			logi << "Created upper bound " << typeid(*upper_bound).name()
					<< endl;

			solver = new AEMS(model, lower_bound, upper_bound);
		} else
			solver = new BeliefBaselineSolver(lower_bound);
	} // POMCP or DPOMCP
	else if (solver_type == "POMCP" || solver_type == "DPOMCP") {
		string ptype = options[E_PRIOR] ? options[E_PRIOR].arg : "DEFAULT";
		POMCPPrior *prior = model->CreatePOMCPPrior(ptype);

		logi << "Created POMCP prior " << typeid(*prior).name() << endl;

		if (options[E_PRUNE]) {
			prior->exploration_constant(Globals::config.pruning_constant);
		}

		if (solver_type == "POMCP")
			solver = new POMCP(model, prior);
		else
			solver = new DPOMCP(model, prior);
	} else { // Unsupported solver
		cerr << "ERROR: Unsupported solver type: " << solver_type << endl;
		exit(1);
	}

	assert(solver != NULL);
	solver->belief(belief);

	return solver;
}

void PlannerBase::OptionParse(option::Option *options, int &num_runs,
		string &world_type, string &belief_type, int &time_limit,
		string &solver_type, bool &search_solver) {
	if (options[E_SILENCE])
		Globals::config.silence = true;

	if (options[E_DEPTH])
		Globals::config.search_depth = atoi(options[E_DEPTH].arg);

	if (options[E_DISCOUNT])
		Globals::config.discount = atof(options[E_DISCOUNT].arg);

	if (options[E_SEED])
		Globals::config.root_seed = atoi(options[E_SEED].arg);
	else { // last 9 digits of current time in milli second
		long millis = (long) get_time_second() * 1000;
		long range = (long) pow((double) 10, (int) 9);
		Globals::config.root_seed = (unsigned int) (millis
				- (millis / range) * range);
	}

	if (options[E_TIMEOUT])
		Globals::config.time_per_move = atof(options[E_TIMEOUT].arg);

	if (options[E_NUMPARTICLES])
		Globals::config.num_scenarios = atoi(options[E_NUMPARTICLES].arg);

	if (options[E_PRUNE])
		Globals::config.pruning_constant = atof(options[E_PRUNE].arg);

	if (options[E_GAP])
		Globals::config.xi = atof(options[E_GAP].arg);

	if (options[E_SIM_LEN])
		Globals::config.sim_len = atoi(options[E_SIM_LEN].arg);

	if (options[E_WORLD])
		world_type = options[E_WORLD].arg;

	if (options[E_MAX_POLICY_SIM_LEN])
		Globals::config.max_policy_sim_len = atoi(
				options[E_MAX_POLICY_SIM_LEN].arg);

	if (options[E_DEFAULT_ACTION])
		Globals::config.default_action = options[E_DEFAULT_ACTION].arg;

	if (options[E_RUNS])
		num_runs = atoi(options[E_RUNS].arg);

	if (options[E_BELIEF])
		belief_type = options[E_BELIEF].arg;

	if (options[E_TIME_LIMIT])
		time_limit = atoi(options[E_TIME_LIMIT].arg);

	if (options[E_NOISE])
		Globals::config.noise = atof(options[E_NOISE].arg);

	search_solver = options[E_SEARCH_SOLVER];

	if (options[E_SOLVER])
		solver_type = options[E_SOLVER].arg;

	if (options[E_GPU])
	{
		Globals::config.useGPU = atoi(options[E_GPU].arg);
		cout<<"[CmdLine] use GPU: "<<Globals::config.useGPU<<endl;
	}

	if (options[E_GPUID])
	{
		Globals::config.GPUid = atoi(options[E_GPUID].arg);
		cout<<"[CmdLine] GPU ID: "<<Globals::config.GPUid<<endl;
	}

	if (options[E_CPU])
	{
		Globals::config.use_multi_thread_ = atoi(options[E_CPU].arg);
		cout<<"[CmdLine] use CPU multi_threading: "<<Globals::config.use_multi_thread_<<endl;
	}

	if (options[E_THREAD_NUM])
	{
		Globals::config.NUM_THREADS = atoi(options[E_THREAD_NUM].arg);
		cout<<"[CmdLine] number of CPU threads: "<<Globals::config.NUM_THREADS<<endl;
	}

	if (options[E_EXPLORATION])
	{
		if(options[E_EXPLORATION].arg=="Vloss")
			Globals::config.exploration_mode = VIRTUAL_LOSS;
		else if(options[E_EXPLORATION].arg=="UCT")
			Globals::config.exploration_mode = UCT;
		cout<<"[CmdLine] exploration scheme (0 virtual loss; 1 UCT bound): "<<Globals::config.exploration_mode<<endl;
	}

	if (options[E_EXPLORE_CONST])
	{
		Globals::config.exploration_constant = atof(options[E_EXPLORE_CONST].arg);
		cout<<"[CmdLine] exploration constant: "<<Globals::config.exploration_constant<<endl;
	}

	if (options[E_EXPLORE_CONST_O])
	{
		Globals::config.exploration_constant_o = atof(options[E_EXPLORE_CONST_O].arg);
		cout<<"[CmdLine] exploration constant for obs branches: "<<Globals::config.exploration_constant_o<<endl;
	}

	if(options[E_DESPOT_THREAD])
	{
		Globals::config.enable_despot_thread = atoi(options[E_DESPOT_THREAD].arg);
		cout<<"[CmdLine] DESPOT thread mode: "<<Globals::config.enable_despot_thread<<endl;
	}

	if(options[E_EXP_MODE])
	{
		Globals::config.experiment_mode = atoi(options[E_EXP_MODE].arg);
		cout<<"[CmdLine] Use experiment mode: " << Globals::config.experiment_mode << endl;
	}

	if(options[E_THREAD_GAP])
	{
		Globals::config.despot_thread_gap = atoi(options[E_THREAD_GAP].arg);
		cout<<"[CmdLine] Number of trials to launch one despot thread: " <<
				Globals::config.despot_thread_gap << endl;
	}
	else{
		Globals::config.despot_thread_gap = Globals::config.NUM_THREADS;
	}

	if(options[E_SWITCH_THRESH])
	{
		Globals::config.expanstion_switch_thresh = atoi(options[E_SWITCH_THRESH].arg);
		cout<<"[CmdLine] Threshold for switching back to CPU expansion: " << Globals::config.expanstion_switch_thresh << endl;
	}



	int verbosity = logging::level();
	if (options[E_VERBOSITY])
		verbosity = atoi(options[E_VERBOSITY].arg);
	logging::level(verbosity);

	cout << "[CmdLine] Using verbosity level " << logging::level() << endl;
}

void PlannerBase::InitializeLogger(Logger *&logger,
		option::Option *options, DSPOMDP *model, Belief* belief, Solver *solver,
		int num_runs, clock_t main_clock_start, World* world, string world_type,
		int time_limit, string solver_type) {

	if (time_limit != -1) {
		logger = new Logger(model, belief, solver, world, world_type,
				main_clock_start, &cout,
				EvalLog::curr_inst_start_time + time_limit,
				num_runs * Globals::config.sim_len);
	} else {
		logger = new Logger(model, belief, solver, world, world_type,
				main_clock_start, &cout);
	}
}

void PlannerBase::DisplayParameters(option::Option *options, DSPOMDP *model) {

	string lbtype = options[E_LBTYPE] ? options[E_LBTYPE].arg : "DEFAULT";
	string ubtype = options[E_UBTYPE] ? options[E_UBTYPE].arg : "DEFAULT";
	default_out<< "[Parameters] Model = " << typeid(*model).name() << endl
	<< "[Parameters] Random root seed = " << Globals::config.root_seed << endl
	<< "[Parameters] Search depth = " << Globals::config.search_depth << endl
	<< "[Parameters] Discount = " << Globals::config.discount << endl
	<< "[Parameters] Simulation steps = " << Globals::config.sim_len << endl
	<< "[Parameters] Number of scenarios = " << Globals::config.num_scenarios
	<< endl
	<< "[Parameters] Search time per step = " << Globals::config.time_per_move
	<< endl
	<< "[Parameters] Regularization constant = "
	<< Globals::config.pruning_constant << endl
	<< "[Parameters] Lower bound = " << lbtype << endl
	<< "[Parameters] Upper bound = " << ubtype << endl
	<< "[Parameters] Policy simulation depth = "
	<< Globals::config.max_policy_sim_len << endl
	<< "[Parameters] Target gap ratio = " << Globals::config.xi << endl;
	// << "[Parameters] Solver = " << typeid(*solver).name() << endl << endl;
}

void PlannerBase::PrintResult(int num_runs, Logger *logger,
		clock_t main_clock_start) {

	logger->PrintStatistics(num_runs);
	cout << "Total time: Real / CPU = "
			<< (get_time_second() - EvalLog::curr_inst_start_time) << " / "
			<< (double(clock() - main_clock_start) / CLOCKS_PER_SEC) << "s"
			<< endl;
}

} // namespace despot
