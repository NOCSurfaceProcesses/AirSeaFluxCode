# Changes

## 1.2.0 (unreleased)

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

