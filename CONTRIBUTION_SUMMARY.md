# Contribution Summary & Public Affairs Release Manifest

**Project:** Reachability Analysis with Sensor Programs
**Author:** Alex Yuan
**Date:** February 4th 2026

> **Distribution Statement:** Approved for public release; distribution is unlimited. Public Affairs release approval #_______.

---

## 1. Overview
This package contains the source code and manuscript associated with the research on high-dimensional reachability analysis with sensor programs. This version includes the "Summer 2025" codebase and the subsequent core library refinements utilized for the final paper results.

## 2. Source Code Manifest
The following files contain original logic or modifications authored by Alex Yuan.

### Core Library Enhancements (`verse/`)
* **`verse/analysis/verifier.py`**: Added `verify_partition` for state-space partitioning.
* **`verse/scenario/scenario.py`**: Integrated partition-based verification logic.
* **`verse/utils/utils.py`**: Updated metrics and performance tracking functions.

### Application & Sensor Logic
* **`demo/aprod/parsed_wrap/`**: Original sensor primitives and input wrappers for sensor parser function.

---

## 3. Benchmark Execution Guide
The following scripts serve as the primary entry points for the benchmarks discussed in the manuscript. All paths are relative to the `demo/aprod/` directory.

| Benchmark | Primary Script(s) | Function |
| :--- | :--- | :--- |
| **Satellite ARPOD** | `orbital_true_multi_switch_ver_gpa_3dep_fix.py` | 3-agent rendezvous ($n=54$) |
| **Robot Rendezvous** | `vis_dubins_rdvz/1c1n_ver_parser.py` | 2-agent coordinate-free robot rendezvous |
| **Highway Merging** | `stanley_controller/m1_1c1n_dryvr.py` | Stanley control merging |

---

## 4. Instructions for Review
To verify the primary results:

1. **Environment Requirements:**
    - **Operating System:** **Linux is highly recommended.** This project relies on `auto_lirpa` (v0.6.0), which has known stability issues on Windows.
    - **Critical Dependencies:** - **PyTorch:** Version **2.3.1** (specified in the Dockerfile).
        - **auto_lirpa:** Version **0.6.0**.
    - **Reference Environment:** A `Dockerfile` is included in the root directory. It is verified for a satellite ARPOD scenario (`orbital_docking_sensor_gpa_parser_wrapper.py`) and serves as the authoritative reference for all system dependencies and environment configuration.
    - **Local Install:** If installing locally, ensure `auto_lirpa` v0.6.0 and PyTorch 2.3.1 are used.

2. **Installation (Required):**
    Before running any benchmarks, you **must install the Verse library** to ensure the `verse` namespace is accessible to the Python interpreter. From the repository root directory, run:
    ```bash
    pip install -e .
    ```
    *(Refer to the main `README.md` for additional troubleshooting regarding the base library installation.)*

3. **Execution:**
    - Navigate to the root demo directory:
      ```bash
      cd demo/aprod
      ```
    - Run the primary satellite benchmark script:
      ```bash
      python3 orbital_true_multi_switch_ver_gpa_3dep_fix.py
      ```

**Note on Execution Modes:**
Scripts are configured for the primary results discussed in the manuscript. To toggle between **Verification** and **Simulation** modes, users may need to selectively comment/uncomment the corresponding execution lines (typically labeled `scenario.verify()` or `scenario.simulate()`) within the `main` block of each script.