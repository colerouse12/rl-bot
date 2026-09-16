#pragma once

// Production training/deployment boundary. Reuse this header and the pinned
// RLGymCPP implementation in a future RLBot adapter; Python BotObs is unrelated.
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rlbot {
inline constexpr const char* kObservationSchema = "rlbot.advanced-1v1.v2";
inline constexpr const char* kActionSchema = "rlbot.default-action-90.v1";
inline constexpr int kObservationSize = 109;
inline constexpr int kActionCount = 90;
inline constexpr int kTickSkip = 8;
inline constexpr int kActionDelay = 7;

inline void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class Observation final : public RLGC::AdvancedObs {
public:
    RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override {
        Require(state.players.size() == 2 && state.players[0].team != state.players[1].team,
                "Observation schema requires exactly one blue and one orange player");
        auto values = AdvancedObs::BuildObs(player, state);
        Require(values.size() == kObservationSize, "Unexpected observation dimension");
        for (float value : values)
            Require(std::isfinite(value), "Observation contains NaN or infinity");
        return values;
    }
};

class Actions final : public RLGC::DefaultAction {
public:
    Actions() {
        Require(GetActionAmount() == kActionCount, "Pinned action table changed");
    }
    RLGC::Action ParseAction(int index, const RLGC::Player& player, const RLGC::GameState& state) override {
        Require(index >= 0 && index < kActionCount, "Policy returned an invalid action index");
        return DefaultAction::ParseAction(index, player, state);
    }
    std::vector<uint8_t> GetActionMask(const RLGC::Player& player, const RLGC::GameState& state) override {
        auto mask = DefaultAction::GetActionMask(player, state);
        Require(mask.size() == kActionCount && std::any_of(mask.begin(), mask.end(), [](uint8_t x) { return x != 0; }),
                "Action mask has no available actions or has the wrong dimension");
        return mask;
    }
};

class EpisodeTimeout final : public RLGC::TerminalCondition {
    float elapsed_ = 0;
    float limit_;
public:
    explicit EpisodeTimeout(float seconds) : limit_(seconds) {}
    void Reset(const RLGC::GameState&) override { elapsed_ = 0; }
    bool IsTerminal(const RLGC::GameState& state) override {
        elapsed_ += state.deltaTime;
        return elapsed_ >= limit_;
    }
    bool IsTruncation() override { return true; }
};

inline nlohmann::json ContractDescription() {
    using nlohmann::json;
    Actions actions;
    json table = json::array();
    for (const auto& action : actions.actions) {
        json row = json::array();
        for (float value : action) row.push_back(value);
        table.push_back(row);
    }
    return {
        {"observation_schema", kObservationSchema}, {"observation_size", kObservationSize},
        {"action_schema", kActionSchema}, {"action_count", kActionCount},
        {"tick_skip", kTickSkip}, {"action_delay", kActionDelay}, {"physics_hz", 120},
        {"observation_builder", "pinned RLGC::AdvancedObs with corrected canonical/inverted boost timer selection"},
        {"position_coefficient", RLGC::AdvancedObs::POS_COEF},
        {"velocity_coefficient", RLGC::AdvancedObs::VEL_COEF},
        {"angular_velocity_coefficient", RLGC::AdvancedObs::ANG_VEL_COEF},
        {"feature_order", json::array({
            "ball position xyz / 5000; velocity xyz / 2300; angular velocity xyz / 3 (9)",
            "previous action: throttle, steer, pitch, yaw, roll, jump, boost, handbrake (8)",
            "34 boost pads in pinned CommonValues order: available=1, otherwise 1/(1+timer_seconds)",
            "self (29), then opponent (29); no teammates in 1v1",
            "each player: position xyz/5000, forward xyz, up xyz, velocity xyz/2300, angular velocity xyz/3, local angular velocity xyz/3, local ball relative position xyz/5000, local ball relative velocity xyz/2300, boost/100, on_ground, has_flip_or_jump, demoed, has_jumped"
        })},
        {"team_inversion", "Orange: negate world x/y of positions, velocities, angular velocities and orientation axes; use inverted boost-pad order. Previous controls stay unchanged. Local vectors use inverted car orientation."},
        {"action_columns", {"throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"}},
        {"action_table", table},
        {"action_masks", {{"ground", actions.groundMask}, {"air", actions.airMask},
                          {"jump", actions.jumpMask}, {"boost", actions.boostMask}}},
        {"action_mask_rule", "Pinned DefaultAction::GetActionMask: ground/air selection, subtract boost at zero boost, add jump when HasFlipOrJump or turtled. Preserves upstream ordering and masks exactly."}
    };
}
} // namespace rlbot
