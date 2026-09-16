"""Apply the project's narrow, reproducible patch to the pinned GigaLearn source.

The ignored checkout stays upstream-owned. This script accepts only pristine files
from the pinned commit or the exact patched result; any other edit fails closed.
Run again safely after fetching dependencies. No network access is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


PIN = "86e11c3881a21225a43c36b14af40f0b3c87e7dd"
PUBLIC = "GigaLearnCPP/src/public/GigaLearnCPP/"
PRIVATE = "GigaLearnCPP/src/private/GigaLearnCPP/"
RLGYM = "GigaLearnCPP/RLGymCPP/src/RLGymCPP/"


def replace(text: str, before: str, after: str, count: int = 1) -> str:
    actual = text.count(before)
    if actual != count:
        raise RuntimeError(f"Upstream patch context mismatch: expected {count}, got {actual}: {before[:100]!r}")
    return text.replace(before, after)


def patch_config(text: str) -> str:
    return replace(text, "\t\tint numGames = 300;", """\t\tint numGames = 300;

\t\t// Project hook: additional PPO iterations per Start(); zero runs indefinitely.
\t\tuint64_t maxIterations = 0;
\t\t// Legacy interactive thread is opt-in and never started for bounded runs.
\t\tbool enableQuitKey = false;""")


def patch_header(text: str) -> str:
    return replace(text, "\t\tStepCallbackFn stepCallback = NULL;", """\t\tStepCallbackFn stepCallback = NULL;

\t\t// Project hooks: validate each completed update before saving, and attach metadata.
\t\tstd::function<void(Learner*, Report&)> iterationCallback;
\t\tstd::function<void(Learner*)> saveCallback;
\t\tstd::function<bool()> stopRequested;""")


def patch_learner(text: str) -> str:
    text = replace(text,
        "Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :\n\tenvCreateFn(envCreateFn), config(config), stepCallback(stepCallback)",
        "Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig inputConfig, StepCallbackFn stepCallback) :\n\tenvCreateFn(envCreateFn), config(inputConfig), stepCallback(stepCallback)")
    text = replace(text, "\tppo->SaveTo(saveFolder);", """\tppo->SaveTo(saveFolder);
\tif (saveCallback)
\t\tsaveCallback(this);""")
    text = replace(text, "auto& prevRewards = envSet->state.lastRewards[i];", "auto& prevRewards = envSet->state.lastRewards[arenaIdx];")
    text = replace(text, "GGL::Learner::~Learner() {\n\tdelete ppo;", """GGL::Learner::~Learner() {
\t// Wait for a possibly outstanding first-half physics step before freeing arenas.
\ttry { envSet->Sync(); } catch (...) {} // The active failure is reported by Start().
\tdelete envSet;
\tdelete returnStat;
\tdelete obsStat;
\tdelete ppo;""")
    # Only the ordinary PPO entry point gets a bounded runner; transfer learning
    # is deliberately outside this project's initial training contract.
    before_start, start = text.split("void GGL::Learner::Start() {", 1)
    start = replace(start, """\t\tbool saveQueued;
\t\tstd::thread keyPressThread;
\t\tStartQuitKeyThread(saveQueued, keyPressThread);""", """\t\tbool saveQueued = false;
\t\tstd::thread keyPressThread;
\t\tif (config.enableQuitKey && config.maxIterations == 0)
\t\t\tStartQuitKeyThread(saveQueued, keyPressThread);
\t\tconst uint64_t initialIterations = totalIterations;
\t\tTORCH_CHECK(!render || config.maxIterations == 0, "Bounded PPO cannot use render mode");""")
    start = replace(start, """\t\t\t\t\t\tif (stepCallback)
\t\t\t\t\t\t\tstepCallback(this, envSet->state.gameStates, report);""", """\t\t\t\t\t\tfor (float reward : envSet->state.rewards)
\t\t\t\t\t\t\tTORCH_CHECK(std::isfinite(reward), "Environment produced a non-finite reward");

\t\t\t\t\t\tif (stepCallback)
\t\t\t\t\t\t\tstepCallback(this, envSet->state.gameStates, report);""")
    start = replace(start,
        'report["Episode Length"] = 1.f / (tTerminals == 1).to(torch::kFloat32).mean().item<float>();',
        'report["Episode Length"] = 1.f / (tTerminals != 0).to(torch::kFloat32).mean().item<float>();')
    start = replace(start, """\t\t\t\tif (saveQueued) {
\t\t\t\t\tif (!config.checkpointFolder.empty())
\t\t\t\t\t\tSave();
\t\t\t\t\texit(0);
\t\t\t\t}

\t\t\t\tif (!config.checkpointFolder.empty()) {
\t\t\t\t\tif (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
\t\t\t\t\t\t// Auto-save
\t\t\t\t\t\tSave();
\t\t\t\t\t}
\t\t\t\t}

\t\t\t\treport.Finish();""", """\t\t\t\t// Validate the completed update before any checkpoint is written.
\t\t\t\treport.Finish();
\t\t\t\tif (iterationCallback)
\t\t\t\t\titerationCallback(this, report);

\t\t\t\tconst bool reachedLimit = (config.maxIterations != 0 &&
\t\t\t\t\ttotalIterations - initialIterations >= config.maxIterations) ||
\t\t\t\t\t(stopRequested && stopRequested());
\t\t\t\tconst bool periodicSave = config.tsPerSave > 0 &&
\t\t\t\t\ttotalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave;
\t\t\t\tif (!config.checkpointFolder.empty() && (saveQueued || reachedLimit || periodicSave))
\t\t\t\t\tSave();
\t\t\t\tif (saveQueued)
\t\t\t\t\texit(0);""")
    start = replace(start, """\t\t\t\t);
\t\t\t}
\t\t}
\t\t
\t} catch (std::exception& e) {""", """\t\t\t\t);
\t\t\t\tif (reachedLimit)
\t\t\t\t\treturn;
\t\t\t}
\t\t}
\t\t
\t} catch (std::exception& e) {""")
    return before_start + "void GGL::Learner::Start() {" + start


def patch_ppo(text: str) -> str:
    text = replace(text, "using namespace torch;", """using namespace torch;

namespace {
\tvoid RequireFinite(const torch::Tensor& tensor, const char* label) {
\t\tTORCH_CHECK(torch::isfinite(tensor).all().item<bool>(), label, " contains NaN/inf");
\t}
}""")
    text = replace(text,
        '\tauto logits = models["policy"]->Forward(obs, halfPrec) / temperature;',
        '\tauto logits = models["policy"]->Forward(obs, halfPrec) / temperature;\n\tRequireFinite(logits, "Policy logits");')
    text = replace(text,
        '\treturn models["critic"]->Forward(obs, config.useHalfPrecision).flatten();',
        '\tauto values = models["critic"]->Forward(obs, config.useHalfPrecision).flatten();\n\tRequireFinite(values, "Critic output");\n\treturn values;')
    text = replace(text, "\t// Save parameters first", "\tuint64_t optimizerSteps = 0;\n\n\t// Save parameters first")
    text = replace(text,
        "\t\t\t\t\tavgRelEntropyLoss += (curEntropy * config.entropyScale) / curPolicyLoss;",
        "\t\t\t\t\tif (std::abs(curPolicyLoss) > 1e-12f)\n\t\t\t\t\t\tavgRelEntropyLoss += (curEntropy * config.entropyScale) / curPolicyLoss;")
    text = replace(text, "\t\t\t\tif (trainPolicy && trainCritic) {", """\t\t\t\tif (trainPolicy)
\t\t\t\t\tRequireFinite(ppoLoss, "PPO loss");
\t\t\t\tif (trainCritic)
\t\t\t\t\tRequireFinite(criticLoss, "Critic loss");

\t\t\t\tif (trainPolicy && trainCritic) {""")
    text = replace(text, "\t\t\tmodels.StepOptims();", """\t\t\tfor (auto* model : models)
\t\t\t\tfor (const auto& parameter : model->parameters())
\t\t\t\t\tif (parameter.grad().defined())
\t\t\t\t\t\tRequireFinite(parameter.grad(), "Model gradient");
\t\t\tmodels.StepOptims();
\t\t\t++optimizerSteps;
\t\t\tfor (auto* model : models)
\t\t\t\tfor (const auto& parameter : model->parameters())
\t\t\t\t\tRequireFinite(parameter, "Updated model parameter");""")
    text = replace(text, """\tif (!isFirstIteration) {
\t\t// These metrics give bad data on the first iteration, which will mess up graph scaling
\t\t// So we'll just skip them for the first iteration""", """\treport["Optimizer Steps"] = optimizerSteps;
\t// Keep first-update measurements for the bounded readiness check as well.
\t{""")
    return text


def patch_envset(text: str) -> str:
    text = replace(text, "\t\tappendMutex.lock();\n\t\t{", "\t\t{\n\t\t\tstd::lock_guard<std::mutex> appendLock(appendMutex);")
    text = replace(text, "\t\tappendMutex.unlock();\n", "")
    return replace(text, "userInfo->arenaIdx = idx;", "userInfo->arenaIdx = static_cast<int>(arenas.size()) - 1;")


def patch_threadpool(text: str) -> str:
    text = replace(text, "#include <thread_pool.h>", "#include <thread_pool.h>\n#include <exception>\n#include <mutex>\n#include <utility>")
    text = replace(text, "\t\tdp::thread_pool<>* _tp;", """\t\tdp::thread_pool<>* _tp;
\t\tstd::mutex _errorMutex;
\t\tstd::exception_ptr _error;""")
    text = replace(text, "\t\t\t_tp->enqueue_detach(func, args...);", """\t\t\tauto job = std::bind(std::forward<Function>(func), std::forward<Args>(args)...);
\t\t\t_tp->enqueue_detach([this, job = std::move(job)]() mutable {
\t\t\t\ttry {
\t\t\t\t\tjob();
\t\t\t\t} catch (...) {
\t\t\t\t\tstd::lock_guard<std::mutex> lock(_errorMutex);
\t\t\t\t\tif (!_error)
\t\t\t\t\t\t_error = std::current_exception();
\t\t\t\t}
\t\t\t});""")
    return replace(text, "\t\t\t_tp->wait_for_tasks();", """\t\t\t_tp->wait_for_tasks();
\t\t\tstd::exception_ptr error;
\t\t\t{
\t\t\t\tstd::lock_guard<std::mutex> lock(_errorMutex);
\t\t\t\terror = std::exchange(_error, {});
\t\t\t}
\t\t\tif (error)
\t\t\t\tstd::rethrow_exception(error);""")


def patch_cmake(text: str) -> str:
    # The trainer and learner must share RocketSim's stage and global thread pool.
    return replace(text, "add_library(GigaLearnCPP SHARED ${FILES_SRC})", "add_library(GigaLearnCPP STATIC ${FILES_SRC})")


def patch_framework(text: str) -> str:
    return replace(text, "#ifdef WITHIN_GGL\n#define RG_IMEXPORT RG_EXPORTED", "#if defined(GGL_STATIC)\n#define RG_IMEXPORT\n#elif defined(WITHIN_GGL)\n#define RG_IMEXPORT RG_EXPORTED")


def patch_gamestate(text: str) -> str:
    # Timer ordering must match GetBoostPads for the same observing team.
    return replace(text,
        "return inverted ? boostPadTimers : boostPadTimersInv;",
        "return inverted ? boostPadTimersInv : boostPadTimers;")


PATCHERS = {
    "GigaLearnCPP/CMakeLists.txt": patch_cmake,
    PUBLIC + "Framework.h": patch_framework,
    PUBLIC + "LearnerConfig.h": patch_config,
    PUBLIC + "Learner.h": patch_header,
    PUBLIC + "Learner.cpp": patch_learner,
    PRIVATE + "PPO/PPOLearner.cpp": patch_ppo,
    RLGYM + "EnvSet/EnvSet.cpp": patch_envset,
    RLGYM + "ThreadPool.h": patch_threadpool,
    RLGYM + "Gamestates/GameState.h": patch_gamestate,
}


def git(source: Path, *args: str) -> str:
    command = ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-C", str(source), *args]
    return subprocess.check_output(command).decode("utf-8-sig").replace("\r\n", "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/gigalearn"))
    parser.add_argument("--check", action="store_true", help="Validate patched files without writing")
    args = parser.parse_args()
    source = args.source.resolve()
    if Path(git(source, "rev-parse", "--show-toplevel").strip()).resolve() != source:
        raise RuntimeError("--source must point at the GigaLearn checkout root")
    if git(source, "rev-parse", "HEAD").strip() != PIN:
        raise RuntimeError(f"GigaLearn must be checked out at {PIN}")
    changed = set(filter(None, git(source, "diff", "--name-only", "--no-renames", "-z", PIN, "--").split("\0")))
    unexpected = changed - PATCHERS.keys()
    # Include ignored files: CMake's recursive glob can compile an ignored .cpp.
    untracked = set(filter(None, git(source, "ls-files", "--others", "-z", "--").split("\0")))
    if unexpected or untracked:
        raise RuntimeError("Unexpected files in the pinned source checkout: " + ", ".join(sorted(unexpected | untracked)))
    pending: list[tuple[Path, str]] = []
    for name, patcher in PATCHERS.items():
        pristine = git(source, "show", f"{PIN}:{name}")
        patched = patcher(pristine)
        path = source / name
        actual = path.read_text(encoding="utf-8-sig")
        if actual == patched:
            continue
        if actual != pristine:
            raise RuntimeError(f"Refusing to overwrite unexpected local changes: {path}")
        pending.append((path, patched))
    # Validate every file before mutating any file.
    if args.check and pending:
        raise RuntimeError(f"{len(pending)} source files still require the project patch")
    for path, patched in pending:
        path.write_text(patched, encoding="utf-8", newline="\n")
    print(f"GigaLearn {PIN[:12]}: {len(pending)} files patched; {len(PATCHERS)} files verified")


if __name__ == "__main__":
    main()
