#include "contract.h"
#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/StateSetters/StateSetter.h>
#include <ATen/Parallel.h>
#include <fstream>
#include <iostream>
#include <random>
#include <optional>
#include <limits>
#include <set>
#include <csignal>
#include <chrono>

using nlohmann::json;
namespace fs = std::filesystem;
using rlbot::Require;

namespace {
constexpr const char* kMetadataFile = "PROJECT_METADATA.json";
volatile std::sig_atomic_t stopRequested = 0;
void RequestStop(int) { stopRequested = 1; }

json ReadJson(const fs::path& path) {
    std::ifstream input(path);
    Require(input.good(), "Cannot read JSON file: " + path.string());
    return json::parse(input);
}

void WriteJson(const fs::path& path, const json& data) {
    std::ofstream output(path, std::ios::trunc);
    Require(output.good(), "Cannot write JSON file: " + path.string());
    output << data.dump(2) << '\n';
    output.flush();
    Require(output.good(), "Writing JSON failed: " + path.string());
}

void ExactKeys(const json& value, const std::set<std::string>& keys, const std::string& location) {
    Require(value.is_object(), location + " must be an object");
    for (const auto& [key, ignored] : value.items())
        Require(keys.count(key), "Unknown configuration field: " + location + "." + key);
    for (const auto& key : keys)
        Require(value.contains(key), "Missing configuration field: " + location + "." + key);
}

int64_t Integer(const json& object, const char* key, int64_t minimum, int64_t maximum = INT32_MAX) {
    const auto& value = object.at(key);
    Require(value.is_number_integer(), std::string(key) + " must be an integer");
    const int64_t number = value.get<int64_t>();
    Require(number >= minimum && number <= maximum, std::string(key) + " is outside the supported range");
    return number;
}

double Number(const json& object, const char* key, double minimum, double maximum) {
    const auto& value = object.at(key);
    Require(value.is_number(), std::string(key) + " must be numeric");
    const auto number = value.get<double>();
    Require(std::isfinite(number) && number >= minimum && number <= maximum,
            std::string(key) + " is outside the supported finite range");
    return number;
}

void ValidateConfig(const json& config) {
    ExactKeys(config, {"schema_version", "run_name", "device", "random_seed", "torch_threads", "num_games", "tick_skip",
        "action_delay", "mesh_dir", "checkpoint_dir", "max_iterations", "timesteps_per_save", "checkpoints_to_keep",
        "environment", "network", "ppo"}, "config");
    Require(Integer(config, "schema_version", 1, 1) == 1, "Unsupported config schema");
    Require(config.at("device") == "cpu", "This validated build supports device=cpu only");
    for (auto key : {"run_name", "mesh_dir", "checkpoint_dir"})
        Require(config.at(key).is_string() && !config.at(key).get<std::string>().empty(), std::string(key) + " must not be empty");
    Integer(config, "random_seed", 0);
    Integer(config, "torch_threads", 1, 64);
    Integer(config, "num_games", 1, 1024);
    Integer(config, "tick_skip", rlbot::kTickSkip, rlbot::kTickSkip);
    Integer(config, "action_delay", rlbot::kActionDelay, rlbot::kActionDelay);
    Integer(config, "max_iterations", 0);
    Integer(config, "timesteps_per_save", 1);
    Integer(config, "checkpoints_to_keep", 1, 1000);
    const auto& env = config.at("environment");
    ExactKeys(env, {"no_touch_seconds", "episode_seconds"}, "environment");
    Number(env, "no_touch_seconds", 0.1, 3600);
    Number(env, "episode_seconds", env.at("no_touch_seconds").get<double>(), 3600);
    const auto& network = config.at("network");
    ExactKeys(network, {"shared_layers", "policy_layers", "critic_layers", "layer_norm"}, "network");
    Require(network.at("layer_norm").is_boolean(), "layer_norm must be boolean");
    for (const auto key : {"shared_layers", "policy_layers", "critic_layers"}) {
        const auto& layers = network.at(key);
        Require(layers.is_array() && !layers.empty() && layers.size() <= 8, std::string(key) + " requires 1..8 layers");
        for (const auto& layer : layers)
            Require(layer.is_number_integer() && layer.get<int64_t>() >= 8 && layer.get<int64_t>() <= 4096,
                    std::string(key) + " has unsupported layer size");
    }
    const auto& ppo = config.at("ppo");
    ExactKeys(ppo, {"timesteps_per_iteration", "batch_size", "minibatch_size", "epochs", "policy_learning_rate",
        "critic_learning_rate", "entropy_scale", "gae_gamma", "gae_lambda"}, "ppo");
    auto rollout = Integer(ppo, "timesteps_per_iteration", 2);
    auto batch = Integer(ppo, "batch_size", 2);
    auto minibatch = Integer(ppo, "minibatch_size", 2);
    Require(rollout >= batch && batch % minibatch == 0, "PPO requires rollout >= batch and batch divisible by minibatch");
    Integer(ppo, "epochs", 1, 100);
    Number(ppo, "policy_learning_rate", 1e-8, 1);
    Number(ppo, "critic_learning_rate", 1e-8, 1);
    Number(ppo, "entropy_scale", 0, 1);
    Number(ppo, "gae_gamma", 0.01, 1);
    Number(ppo, "gae_lambda", 0, 1);
}

json Sources() {
    return {{"gigalearn_commit", RLBOT_GIGALEARN_COMMIT}, {"rlgymcpp_tree", RLBOT_RLGYMCPP_TREE},
            {"rocketsim_tree", RLBOT_ROCKETSIM_TREE}, {"pybind11_tree", RLBOT_PYBIND11_TREE},
            {"runtime_patch_schema", "rlbot.runtime.v2"}};
}

json Compatibility(const json& config) {
    return {{"contract", rlbot::ContractDescription()}, {"network", config.at("network")},
            {"sources", Sources()}, {"optimizer", "Adam"}, {"activation", "ReLU"},
            {"libtorch", RLBOT_TORCH_VERSION},
            {"standardize_observations", false}, {"standardize_returns", true}};
}

std::optional<fs::path> LatestCheckpoint(const fs::path& folder) {
    if (!fs::exists(folder)) return std::nullopt;
    Require(fs::is_directory(folder), "Checkpoint path is not a directory");
    std::optional<fs::path> latest;
    uint64_t latestSteps = 0;
    for (const auto& entry : fs::directory_iterator(folder)) {
        const auto name = entry.path().filename().string();
        if (!entry.is_directory() || name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c); })) continue;
        const auto steps = std::stoull(name);
        Require(name == std::to_string(steps), "Checkpoint directory must use canonical integer name: " + name);
        if (!latest || steps > latestSteps) { latest = entry.path(); latestSteps = steps; }
    }
    return latest;
}

json ValidateResume(const fs::path& path, const json& config) {
    for (auto filename : {kMetadataFile, "RUNNING_STATS.json", "POLICY.lt", "POLICY_optim.lt", "CRITIC.lt", "CRITIC_optim.lt", "SHARED_HEAD.lt", "SHARED_HEAD_optim.lt"}) {
        // Upstream uppercases the entire model filename, including OPTIM.
        std::string actual = filename;
        if (actual.find("_optim") != std::string::npos) actual.replace(actual.find("_optim"), 6, "_OPTIM");
        Require(fs::is_regular_file(path / actual) && fs::file_size(path / actual) > 0,
                "Incomplete checkpoint: missing or empty " + (path / actual).string());
    }
    const auto metadata = ReadJson(path / kMetadataFile);
    Require(metadata.at("metadata_schema") == 1, "Unsupported checkpoint metadata schema");
    Require(metadata.at("compatibility") == Compatibility(config), "Checkpoint observation/action/network/source contract mismatch");
    const auto stats = ReadJson(path / "RUNNING_STATS.json");
    Require(metadata.at("total_timesteps") == stats.at("total_timesteps") &&
            metadata.at("total_iterations") == stats.at("total_iterations"), "Checkpoint metadata and running stats disagree");
    Require(stats.at("total_timesteps").get<uint64_t>() == std::stoull(path.filename().string()), "Checkpoint directory and timestep counter disagree");
    return metadata;
}

class SeededKickoff final : public RLGC::StateSetter {
    std::mt19937 random_;
public:
    explicit SeededKickoff(uint32_t seed) : random_(seed) {}
    void ResetArena(RocketSim::Arena* arena) override {
        arena->ResetToRandomKickoff(static_cast<int>(random_() & 0x7fffffff));
    }
};

class FiniteReward final : public RLGC::Reward {
    std::unique_ptr<RLGC::Reward> inner_;
public:
    explicit FiniteReward(RLGC::Reward* inner) : inner_(inner) {}
    void Reset(const RLGC::GameState& state) override { inner_->Reset(state); }
    void PreStep(const RLGC::GameState& state) override { inner_->PreStep(state); }
    std::vector<float> GetAllRewards(const RLGC::GameState& state, bool final) override {
        auto values = inner_->GetAllRewards(state, final);
        Require(values.size() == state.players.size(), "Reward returned incorrect player count");
        for (auto value : values) Require(std::isfinite(value), "Nonfinite reward: " + inner_->GetName());
        return values;
    }
    std::string GetName() override { return inner_->GetName(); }
};

json RewardDescription() {
    return {{"AirReward", 0.25}, {"FaceBallReward", 0.25}, {"VelocityPlayerToBallReward", 4.0},
            {"StrongTouchReward(20,100)", 60.0}, {"ZeroSum(VelocityBallToGoalReward,1)", 2.0},
            {"PickupBoostReward", 10.0}, {"SaveBoostReward", 0.2},
            {"ZeroSum(BumpReward,0.5)", 20.0}, {"ZeroSum(DemoReward,0.5)", 80.0}, {"GoalReward", 150.0}};
}

RLGC::EnvCreateFn EnvironmentFactory(const json& config) {
    const float noTouch = config.at("environment").at("no_touch_seconds");
    const float timeout = config.at("environment").at("episode_seconds");
    const uint32_t seed = config.at("random_seed");
    return [=](int index) {
        using namespace RLGC;
        EnvCreateResult result = {};
        result.arena = Arena::Create(GameMode::SOCCAR);
        result.arena->AddCar(Team::BLUE);
        result.arena->AddCar(Team::ORANGE);
        result.actionParser = new rlbot::Actions();
        result.obsBuilder = new rlbot::Observation();
        result.stateSetter = new SeededKickoff(seed + static_cast<uint32_t>(index));
        result.terminalConditions = {new GoalScoreCondition(), new NoTouchCondition(noTouch), new rlbot::EpisodeTimeout(timeout)};
        result.rewards = {
            {new FiniteReward(new AirReward()), 0.25f},
            {new FiniteReward(new FaceBallReward()), 0.25f},
            {new FiniteReward(new VelocityPlayerToBallReward()), 4.f},
            {new FiniteReward(new StrongTouchReward(20, 100)), 60.f},
            {new FiniteReward(new ZeroSumReward(new VelocityBallToGoalReward(), 1)), 2.f},
            {new FiniteReward(new PickupBoostReward()), 10.f},
            {new FiniteReward(new SaveBoostReward()), 0.2f},
            {new FiniteReward(new ZeroSumReward(new BumpReward(), 0.5f)), 20.f},
            {new FiniteReward(new ZeroSumReward(new DemoReward(), 0.5f)), 80.f},
            {new FiniteReward(new GoalReward()), 150.f}
        };
        return result;
    };
}

GGL::LearnerConfig LearnerConfig(const json& config) {
    GGL::LearnerConfig result;
    result.deviceType = GGL::LearnerDeviceType::CPU;
    result.numGames = config.at("num_games");
    result.tickSkip = rlbot::kTickSkip;
    result.actionDelay = rlbot::kActionDelay;
    result.randomSeed = config.at("random_seed");
    result.maxIterations = config.at("max_iterations");
    result.enableQuitKey = false;
    result.checkpointFolder = config.at("checkpoint_dir").get<std::string>();
    result.tsPerSave = config.at("timesteps_per_save");
    result.checkpointsToKeep = config.at("checkpoints_to_keep");
    result.sendMetrics = false;
    result.renderMode = false;
    result.standardizeObs = false;
    result.standardizeReturns = true;
    result.savePolicyVersions = false;
    result.trainAgainstOldVersions = false;
    result.skillTracker.enabled = false;
    const auto& ppo = config.at("ppo");
    result.ppo.tsPerItr = ppo.at("timesteps_per_iteration");
    result.ppo.batchSize = ppo.at("batch_size");
    result.ppo.miniBatchSize = ppo.at("minibatch_size");
    result.ppo.epochs = ppo.at("epochs");
    result.ppo.policyLR = ppo.at("policy_learning_rate");
    result.ppo.criticLR = ppo.at("critic_learning_rate");
    result.ppo.entropyScale = ppo.at("entropy_scale");
    result.ppo.gaeGamma = ppo.at("gae_gamma");
    result.ppo.gaeLambda = ppo.at("gae_lambda");
    result.ppo.maxEpisodeDuration = config.at("environment").at("episode_seconds");
    result.ppo.useHalfPrecision = false;
    result.ppo.deterministic = false;
    result.ppo.sharedHead.layerSizes = config.at("network").at("shared_layers").get<std::vector<int>>();
    result.ppo.policy.layerSizes = config.at("network").at("policy_layers").get<std::vector<int>>();
    result.ppo.critic.layerSizes = config.at("network").at("critic_layers").get<std::vector<int>>();
    for (auto* model : {&result.ppo.sharedHead, &result.ppo.policy, &result.ppo.critic}) {
        model->optimType = GGL::ModelOptimType::ADAM;
        model->activationType = GGL::ModelActivationType::RELU;
        model->addLayerNorm = config.at("network").at("layer_norm");
    }
    return result;
}

json ModelState(GGL::Learner* learner, bool requireOptimizer) {
    torch::NoGradGuard noGrad;
    json result = json::object();
    for (auto* model : learner->ppo->models) {
        json shapes = json::array();
        double sum = 0, squaredSum = 0;
        int64_t minimumStep = INT64_MAX;
        for (const auto& parameter : model->parameters()) {
            Require(torch::isfinite(parameter).all().item<bool>(), "Model has nonfinite parameters");
            shapes.push_back(parameter.sizes().vec());
            sum += parameter.to(torch::kFloat64).sum().item<double>();
            squaredSum += parameter.to(torch::kFloat64).square().sum().item<double>();
            if (requireOptimizer) {
                const auto& states = model->optim->state();
                const auto found = states.find(parameter.unsafeGetTensorImpl());
                Require(found != states.end(), "Checkpoint optimizer has no state for a parameter");
                const auto* state = dynamic_cast<const torch::optim::AdamParamState*>(found->second.get());
                Require(state != nullptr && state->step() > 0, "Missing or invalid Adam optimizer step");
                Require(state->exp_avg().sizes() == parameter.sizes() && state->exp_avg_sq().sizes() == parameter.sizes(),
                        "Adam optimizer shape does not match model");
                Require(torch::isfinite(state->exp_avg()).all().item<bool>() && torch::isfinite(state->exp_avg_sq()).all().item<bool>(),
                        "Adam optimizer contains nonfinite values");
                minimumStep = std::min(minimumStep, state->step());
            }
        }
        result[model->modelName] = {{"parameter_shapes", shapes}, {"parameter_sum", sum}, {"parameter_squared_sum", squaredSum},
                                   {"minimum_optimizer_step", requireOptimizer ? minimumStep : 0}};
    }
    return result;
}

void ContractCheck() {
    using namespace RLGC;
    rlbot::Observation builder;
    rlbot::Actions actions;
    GameState state;
    state.players.resize(2);
    for (int i = 0; i < 2; ++i) {
        auto& player = state.players[i];
        player.carId = i + 1;
        player.team = i == 0 ? Team::BLUE : Team::ORANGE;
        player.index = i;
        player.ballTouchedStep = false;
        player.ballTouchedTick = false;
        player.pos = Vec(100 + i * 400, -1500 + i * 2800, 18);
        player.vel = Vec(30 + i * 40, -200 + i * 300, 0);
        player.angVel = Vec(0.2f, -0.1f, 0.3f);
        player.prevAction = {1, -1, 0, -1, 0, 0, 0, 1};
    }
    state.ball.pos = Vec(400, 600, 150);
    state.ball.vel = Vec(100, -300, 10);
    state.ball.angVel = Vec(1, -2, 0.5f);
    state.boostPads[3] = false;
    state.boostPadTimers[3] = 4;
    state.boostPads[9] = false;
    state.boostPadTimers[9] = 2;
    state.boostPadsInv[3] = false;
    state.boostPadTimersInv[3] = 8;
    state.boostPadsInv[9] = false;
    state.boostPadTimersInv[9] = 5;
    auto expected = builder.BuildObs(state.players[0], state);
    const auto orangeExpected = builder.BuildObs(state.players[1], state);
    Require(expected.size() == 109 && std::abs(expected[0] - 0.08f) < 1e-6f, "Observation feature order changed");
    Require(std::abs(expected[17 + 3] - 0.2f) < 1e-6f,
            "Boost timer feature changed: feature=" + std::to_string(expected[20]) +
            " available=" + std::to_string(static_cast<bool>(state.boostPads[3])) +
            " timer=" + std::to_string(state.boostPadTimers[3]) +
            " team=" + std::to_string(static_cast<int>(state.players[0].team)));
    Require(std::abs(expected[51] - 0.02f) < 1e-6f, "Self feature offset changed");
    Require(std::abs(expected[80] - 0.1f) < 1e-6f, "Opponent feature offset changed");
    Require(std::abs(expected[17 + 9] - 1.f / 3.f) < 1e-6f, "Blue boost timer feature differs from its canonical cooldown");
    Require(std::abs(orangeExpected[17 + 3] - 1.f / 9.f) < 1e-6f &&
            std::abs(orangeExpected[17 + 9] - 1.f / 6.f) < 1e-6f,
            "Orange boost timer feature differs from its inverted cooldown");
    GameState inverted = state;
    static_cast<PhysState&>(inverted.ball) = InvertPhys(state.ball);
    inverted.boostPadsInv = state.boostPads;
    inverted.boostPadTimersInv = state.boostPadTimers;
    inverted.boostPads = state.boostPadsInv;
    inverted.boostPadTimers = state.boostPadTimersInv;
    for (auto& player : inverted.players) {
        static_cast<PhysState&>(player) = InvertPhys(player);
        player.team = RS_OPPOSITE_TEAM(player.team);
    }
    const auto actual = builder.BuildObs(inverted.players[0], inverted);
    for (size_t i = 0; i < actual.size(); ++i)
        Require(std::abs(actual[i] - expected[i]) < 1e-6f, "Blue/orange observation inversion differs at " + std::to_string(i));
    const auto orangeActual = builder.BuildObs(inverted.players[1], inverted);
    for (size_t i = 0; i < orangeActual.size(); ++i)
        Require(std::abs(orangeActual[i] - orangeExpected[i]) < 1e-6f, "Orange/blue observation inversion differs at " + std::to_string(i));
    for (auto& player : state.players) {
        for (bool ground : {true, false}) {
            player.isOnGround = ground;
            player.boost = 0;
            const auto mask = actions.GetActionMask(player, state);
            for (int i = 0; i < rlbot::kActionCount; ++i) {
                const auto action = actions.ParseAction(i, player, state);
                for (const auto value : action) Require(std::isfinite(value) && std::abs(value) <= 1, "Invalid control value");
                // Jump actions are added after zero-boost masking by the pinned upstream parser.
                if (mask[i] && !action.jump) Require(action.boost == 0, "Unexpected non-jump boost action at zero boost");
            }
        }
    }
    bool invalidRejected = false;
    try { actions.ParseAction(90, state.players[0], state); } catch (const std::exception&) { invalidRejected = true; }
    Require(invalidRejected, "Invalid action index was not rejected");
    state.ball.pos.x = std::numeric_limits<float>::quiet_NaN();
    bool nonfiniteRejected = false;
    try { builder.BuildObs(state.players[0], state); } catch (const std::exception&) { nonfiniteRejected = true; }
    Require(nonfiniteRejected, "Nonfinite observation was not rejected");
    std::cout << "CONTRACT_CHECK_PASSED obs=109 actions=90 team_inversion=pass masks=pass finite_guard=pass\n";
}

void CheckEnvironment(const json& config) {
    using namespace RLGC;
    EnvSetConfig cfg{};
    cfg.envCreateFn = EnvironmentFactory(config);
    cfg.numArenas = 1;
    cfg.tickSkip = rlbot::kTickSkip;
    cfg.actionDelay = rlbot::kActionDelay;
    cfg.saveRewards = true;
    EnvSet env(cfg);
    Require(env.state.numPlayers == 2 && env.obsSize == 109, "1v1 environment dimensions failed");
    Require(env.state.gameStates[0].ball.pos.Length() < 100, "Kickoff ball is not centered");
    int neutral = -1;
    rlbot::Actions actions;
    for (int i = 0; i < actions.GetActionAmount(); ++i) {
        bool zero = true;
        for (auto value : actions.actions[i]) if (value != 0) zero = false;
        if (zero) neutral = i;
    }
    Require(neutral >= 0, "Neutral action missing");
    auto* arena = env.arenas[0];
    auto* firstCar = *arena->_cars.begin();
    int forward = -1;
    for (int i = 0; i < actions.GetActionAmount(); ++i) {
        const auto& action = actions.actions[i];
        if (action.throttle == 1 && action.steer == 0 && action.boost == 0 && action.jump == 0 && action.handbrake == 0)
            forward = i;
    }
    Require(forward >= 0 && firstCar->controls.throttle == 0, "Kickoff controls or forward action are incorrect");
    const auto startTick = arena->tickCount;
    env.StepFirstHalf(false);
    Require(arena->tickCount == startTick + 7, "Action delay did not advance seven physics ticks");
    Require(firstCar->controls.throttle == 0, "New controls were applied before the seven-tick delay");
    env.StepSecondHalf({forward, forward}, false);
    Require(arena->tickCount == startTick + 8, "Control step did not advance eight physics ticks");
    Require(firstCar->controls.throttle == 1, "New controls were not applied for the final tick");
    Require(std::abs(env.state.gameStates[0].deltaTime - 8.f / 120.f) < 1e-5, "Environment timestep is incorrect");
    for (auto value : env.state.rewards) Require(std::isfinite(value), "Environment reward is nonfinite");
    for (int step = 0; step < 60000 && !env.state.terminals[0]; ++step) {
        env.StepFirstHalf(false);
        env.StepSecondHalf({neutral, neutral}, false);
    }
    Require(env.state.terminals[0] == TerminalType::TRUNCATED, "No-touch condition did not truncate episode");
    env.Reset();
    Require(!env.state.terminals[0] && env.state.gameStates[0].ball.pos.Length() < 100, "Kickoff reset failed");
    auto ball = arena->ball->GetState();
    ball.pos = Vec(0, 5300, 100);
    ball.vel = Vec(0, 500, 0);
    arena->ball->SetState(ball);
    env.StepFirstHalf(false);
    env.StepSecondHalf({neutral, neutral}, false);
    Require(env.state.terminals[0] == TerminalType::NORMAL && env.state.gameStates[0].goalScored,
            "Goal did not terminate the environment normally");
    GoalReward goal;
    for (auto& player : env.state.gameStates[0].players) {
        const float expected = player.team == Team::BLUE ? 1.f : -1.f;
        Require(goal.GetReward(player, env.state.gameStates[0], true) == expected, "Goal reward has wrong sign");
    }
    rlbot::EpisodeTimeout timeout(0.1f);
    GameState synthetic;
    synthetic.deltaTime = 0.06f;
    Require(!timeout.IsTerminal(synthetic) && timeout.IsTerminal(synthetic) && timeout.IsTruncation(), "Episode timeout failed");
    timeout.Reset(synthetic);
    Require(!timeout.IsTerminal(synthetic), "Episode timeout reset failed");
    ball = RocketSim::BallState();
    ball.pos = Vec(0, 0, 500);
    ball.vel = Vec(0, 0, -1000);
    arena->ball->SetState(ball);
    arena->Step(60);
    auto bounced = arena->ball->GetState();
    Require(bounced.pos.z > 80 && bounced.vel.z > 0, "Ball did not bounce from the soccar floor");
    ball.pos = Vec(3900, 0, 300);
    ball.vel = Vec(1000, 0, 0);
    arena->ball->SetState(ball);
    arena->Step(60);
    bounced = arena->ball->GetState();
    Require(bounced.pos.x < 4006 && bounced.vel.x < 0, "Ball did not bounce from the soccar side wall");
    std::cout << "ENVIRONMENT_CHECK_PASSED kickoff=pass tick_skip=8 action_delay=7 no_touch=pass goal=pass timeout=pass floor=pass wall=pass\n";
}

void CheckMeshes(const fs::path& path) {
    const auto soccar = path / "soccar";
    Require(fs::is_directory(soccar), "Soccar collision meshes missing. Pass --mesh-dir PATH containing soccar/*.cmf");
    size_t count = 0;
    for (const auto& entry : fs::directory_iterator(soccar))
        if (entry.is_regular_file() && entry.path().extension() == ".cmf" && entry.file_size() > 0) ++count;
    Require(count >= 16, "Incomplete soccar collision mesh set: expected at least 16 nonempty .cmf files");
}
} // namespace

int main(int argc, char** argv) {
    try {
        fs::path configPath = "configs/1v1-smoke.json";
        bool resume = false, contractCheck = false, environmentCheck = false, validateOnly = false;
        std::optional<int64_t> maxSeconds;
        std::optional<int64_t> targetTimesteps;
        json overrides = json::object();
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "rl-bot-train --config FILE [--resume] [--checkpoint-dir PATH] [--mesh-dir PATH]\n"
                             "  [--seed N] [--device cpu] [--max-iterations N] [--max-seconds N] [--contract-check]\n"
                             "  [--target-timesteps N] [--check-environment] [--validate-config]\n"
                             "Paths are relative to the current directory. max-iterations=0 runs until Ctrl+C.\n"
                             "Ctrl+C saves and exits after the current PPO iteration finishes.\n"
                             "max-seconds is a positive training duration; the current PPO iteration finishes before saving.\n"
                             "target-timesteps is a positive cumulative total, including resumed training; the final iteration may overshoot.\n"
                             "A bounded run saves after N additional PPO iterations; rollout length can overshoot.\n";
                return 0;
            }
            if (option == "--resume") { resume = true; continue; }
            if (option == "--contract-check") { contractCheck = true; continue; }
            if (option == "--check-environment") { environmentCheck = true; continue; }
            if (option == "--validate-config") { validateOnly = true; continue; }
            Require(i + 1 < argc, "Missing argument for " + option);
            const std::string value = argv[++i];
            if (option == "--config") configPath = value;
            else if (option == "--checkpoint-dir") overrides["checkpoint_dir"] = value;
            else if (option == "--mesh-dir") overrides["mesh_dir"] = value;
            else if (option == "--device") overrides["device"] = value;
            else if (option == "--seed" || option == "--max-iterations" || option == "--max-seconds" || option == "--target-timesteps") {
                size_t consumed = 0;
                auto integer = std::stoll(value, &consumed);
                Require(consumed == value.size(), "Invalid integer for " + option);
                if (option == "--max-seconds") {
                    Require(integer > 0 && integer <= INT32_MAX, "--max-seconds must be a positive integer no greater than 2147483647");
                    maxSeconds = integer;
                } else if (option == "--target-timesteps") {
                    Require(integer > 0, "--target-timesteps must be a positive signed 64-bit integer");
                    targetTimesteps = integer;
                } else overrides[option == "--seed" ? "random_seed" : "max_iterations"] = integer;
            } else throw std::runtime_error("Unknown argument: " + option);
        }
        auto config = ReadJson(configPath);
        config.update(overrides);
        ValidateConfig(config);
        if (contractCheck) { ContractCheck(); return 0; }
        if (validateOnly) {
            std::cout << json({{"config", config}, {"compatibility", Compatibility(config)}, {"rewards", RewardDescription()}}).dump(2) << '\n';
            return 0;
        }
        config["mesh_dir"] = fs::absolute(config.at("mesh_dir").get<std::string>()).lexically_normal().string();
        config["checkpoint_dir"] = fs::absolute(config.at("checkpoint_dir").get<std::string>()).lexically_normal().string();
        std::optional<json> loadedMetadata;
        if (!environmentCheck) {
            const auto latest = LatestCheckpoint(config.at("checkpoint_dir").get<std::string>());
            Require(resume || !latest, "Checkpoints already exist. Use --resume or choose a fresh --checkpoint-dir.");
            Require(!resume || latest.has_value(), "--resume requires a saved checkpoint");
            if (latest) loadedMetadata = ValidateResume(*latest, config);
            if (targetTimesteps) {
                const uint64_t savedTimesteps = loadedMetadata ? loadedMetadata->at("total_timesteps").get<uint64_t>() : 0;
                Require(static_cast<uint64_t>(*targetTimesteps) > savedTimesteps,
                        "--target-timesteps has already been reached: checkpoint total=" + std::to_string(savedTimesteps) +
                        ", requested target=" + std::to_string(*targetTimesteps));
            }
        }
        CheckMeshes(config.at("mesh_dir").get<std::string>());
        at::set_num_threads(config.at("torch_threads").get<int>());
        at::set_num_interop_threads(1);
        std::srand(config.at("random_seed").get<unsigned>());
        RocketSim::Math::GetRandEngine().seed(config.at("random_seed").get<unsigned>());
        RocketSim::Init(config.at("mesh_dir").get<std::string>());
        if (environmentCheck) { CheckEnvironment(config); return 0; }
        const auto cfg = LearnerConfig(config);
        auto stepCallback = [](GGL::Learner* learner, const std::vector<RLGC::GameState>& states, GGL::Report& report) {
            for (const float reward : learner->envSet->state.rewards) Require(std::isfinite(reward), "Combined reward is nonfinite");
            for (size_t arenaIndex = 0; arenaIndex < states.size(); ++arenaIndex) {
                const auto& state = states[arenaIndex];
                const bool episodeCompleted = learner->envSet->state.terminals[arenaIndex] != 0;
                report.Add("Game/Goals", state.goalScored ? 1.0 : 0.0);
                report.Add("Game/Timeouts", episodeCompleted && !state.goalScored ? 1.0 : 0.0);
                report.Add("Game/Completed Episodes", episodeCompleted ? 1.0 : 0.0);
                report.Add("Game/Simulated Seconds", state.deltaTime);
                report.AddAvg("Game/Goal", state.goalScored ? 1.0 : 0.0);
                for (const auto& player : state.players) report.AddAvg("Player/Ball Touch", player.ballTouchedStep ? 1.0 : 0.0);
            }
        };
        auto learner = std::make_unique<GGL::Learner>(EnvironmentFactory(config), cfg, stepCallback);
        std::signal(SIGINT, RequestStop);
        std::signal(SIGTERM, RequestStop);
        std::chrono::steady_clock::time_point trainingStarted;
        auto elapsedTrainingSeconds = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - trainingStarted).count();
        };
        auto timeLimitReached = [&] { return maxSeconds && elapsedTrainingSeconds() >= *maxSeconds; };
        auto timestepTargetReached = [&] {
            return targetTimesteps && learner->totalTimesteps >= static_cast<uint64_t>(*targetTimesteps);
        };
        learner->stopRequested = [&] { return stopRequested != 0 || timestepTargetReached() || timeLimitReached(); };
        Require(learner->obsSize == rlbot::kObservationSize && learner->numActions == rlbot::kActionCount, "Learner contract dimensions do not match");
        const auto initialTimesteps = learner->totalTimesteps;
        const auto initialIterations = learner->totalIterations;
        if (loadedMetadata) {
            Require(ModelState(learner.get(), true) == loadedMetadata->at("models"), "Restored model/optimizer state differs from saved metadata");
            std::cout << "RESUME_VERIFIED total_timesteps=" << initialTimesteps << " total_iterations=" << initialIterations << '\n';
        } else ModelState(learner.get(), false);
        const auto metricsPath = cfg.checkpointFolder / "METRICS.jsonl";
        fs::create_directories(cfg.checkpointFolder);
        std::ofstream metricsOutput(metricsPath, resume ? std::ios::app : std::ios::trunc);
        Require(metricsOutput.good(), "Cannot write training metrics: " + metricsPath.string());
        const json durationLimit = maxSeconds ? json(*maxSeconds) : json(nullptr);
        const json timestepTarget = targetTimesteps ? json(*targetTimesteps) : json(nullptr);
        json latestReport = json::object();
        learner->iterationCallback = [&](GGL::Learner* current, GGL::Report& report) {
            for (const auto& [name, value] : report.data) Require(std::isfinite(value), "Nonfinite training metric: " + name);
            for (const auto* metric : {"Policy Update Magnitude", "Critic Update Magnitude"})
                Require(report.Has(metric) && report[metric] > 0, std::string("No optimizer update detected: ") + metric);
            Require(report.Has("Optimizer Steps") && report["Optimizer Steps"] > 0, "No optimizer step was completed");
            ModelState(current, true);
            latestReport = report.data;
            metricsOutput << json({
                {"metrics_schema", 1}, {"elapsed_training_seconds", elapsedTrainingSeconds()},
                {"duration_limit_seconds", durationLimit}, {"total_timesteps", current->totalTimesteps},
                {"target_timesteps", timestepTarget},
                {"total_iterations", current->totalIterations},
                {"session_timesteps", current->totalTimesteps - initialTimesteps},
                {"session_iterations", current->totalIterations - initialIterations},
                {"resumed_from_timesteps", initialTimesteps}, {"metrics", latestReport}
            }).dump() << '\n';
            metricsOutput.flush();
            Require(metricsOutput.good(), "Writing training metrics failed: " + metricsPath.string());
            std::cout << "OPTIMIZER_UPDATE_VERIFIED total_timesteps=" << current->totalTimesteps
                      << " total_iterations=" << current->totalIterations << '\n' << std::flush;
        };
        learner->saveCallback = [&](GGL::Learner* current) {
            const auto path = current->config.checkpointFolder / std::to_string(current->totalTimesteps);
            const bool boundedEnd = cfg.maxIterations > 0 && current->totalIterations - initialIterations >= cfg.maxIterations;
            json metadata = {
                {"metadata_schema", 1}, {"compatibility", Compatibility(config)}, {"run_config", config},
                {"rewards", RewardDescription()}, {"total_timesteps", current->totalTimesteps},
                {"total_iterations", current->totalIterations}, {"models", ModelState(current, true)},
                {"save_reason", stopRequested ? "interrupt-requested" : timestepTargetReached() ? "timestep-target" : timeLimitReached() ? "time-limit" : boundedEnd ? "bounded-run-limit" : "periodic-save"},
                {"elapsed_training_seconds", elapsedTrainingSeconds()}, {"duration_limit_seconds", durationLimit},
                {"target_timesteps", timestepTarget},
                {"resumed_from_timesteps", initialTimesteps}, {"metrics", latestReport},
                {"build", {{"type", RLBOT_BUILD_TYPE}, {"compiler", RLBOT_COMPILER}, {"libtorch", RLBOT_TORCH_VERSION}}},
                {"runtime", {{"device", "cpu"}, {"send_metrics", false}, {"render", false}, {"half_precision", false},
                             {"standardize_returns", true}, {"standardize_observations", false}}},
                {"seed_scope", "Torch sampling, experience shuffle and per-environment kickoff RNG; resume restores models, Adam and running stats, not simulator or RNG trajectory."}
            };
            WriteJson(path / kMetadataFile, metadata);
            std::cout << "CHECKPOINT_METADATA_SAVED " << (path / kMetadataFile).string() << '\n';
        };
        trainingStarted = std::chrono::steady_clock::now();
        learner->Start();
        Require(learner->totalTimesteps > initialTimesteps && learner->totalIterations > initialIterations, "No training update completed");
        std::cout << "TRAINING_RUN_COMPLETED total_timesteps=" << learner->totalTimesteps
                  << " total_iterations=" << learner->totalIterations << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "rl-bot-train: " << exception.what() << '\n';
        return 1;
    }
}
