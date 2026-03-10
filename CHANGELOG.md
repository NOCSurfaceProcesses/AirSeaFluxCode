# Changes

## 1.3.3 (2026-03-10)

Contributors to this release: Joseph Siddons (@jtsiddons).

### Bug Fixes

* Correct conversion of SST to Kelvin, use correct variable (#29).

## 1.3.2 (2026-02-06)

Contributors to this release: Joseph Siddons (@jtsiddons).

### Bug Fixes

* Account for immutability of `pandas>=3.0.0` objects when converted to `numpy` in
  `util_subs.validate_kelvin` (#27)

### Internal Changes

* Update License to include 2026 (#28)

## 1.3.1 (2025-10-30)

Contributors to this release: Joseph Siddons (@jtsiddons).

### Internal Changes

* Corrected classifier for the License in pyproject.toml (#23)

## 1.3.0 (2025-10-30)

Contributors to this release: Joseph Siddons (@jtsiddons), Richard Cornes (@rcornes).

### Breaking Changes

* Most functions now assume temperature inputs are in Kelvin, unless otherwise described by the
  documentation. A warning is displayed if potential Celsius values are detected, but the
  computation will continue without conversion (#21).
* An optional convert switch is added to AirSeaFluxCode.AirSeaFluxCode to allow for conversion of
  possible Celsius values (#21).

### New features and enhancements

* Added height adjustment functions (#13).

### Bug Fixes

* Fixes bug from usage of `numpy.empty` as a final output value when humidity is not set (#16).
* Fixes bug with conversion of temperatures (#21).
* Fixes possible double conversion to Celsius in `AirSeaFluxCode.cs_wl_subs.coolskin.cs` (#21).

### Internal Changes

* Reformatted code (#13).
* Updated License to Apache v2.0 (#12).

## 1.2.0 (2025-03-11)

Contributors to this release: Joseph Siddons (@jtsiddons), Richard Cornes (@rcornes).

### Breaking Changes

* Support dropped for python < 3.9 ([#6](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/6)).

### Internal Changes

* Project has been restructured to match python project standard ([#6](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/6)).

## 1.1.0 (2024-04-17)

* Contributors to this release: Joseph Siddons (@jtsiddons), Richard Cornes (@rcornes), Steven Chan (@SCChan21).

### New features and enhancements

* Minor refactors to simplify `hum_subs.VapourPressure` ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).

### Bug fixes

* Humidity units are now consistently `g/kg` throughout ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).
* Correct function call to `delta` in `cs_wl_subs.cs` ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).
* Input RH values < 1%, and humidity units `g/kg` < 1 now display warning rather than raising error ([#3](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/3)).

### Internal Changes

* Formatting of variables, types, and units are now consistent in documentation ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).
