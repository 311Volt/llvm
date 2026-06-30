//===----------- device.hpp - LLVM Offload Adapter  -----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
#include <OffloadAPI.h>
#include <mutex>
#include <unified-runtime/ur_api.h>

struct ur_device_handle_t_ : ur::offload::handle_base {
  ur_device_handle_t_(ur_platform_handle_t Platform,
                      ol_device_handle_t OffloadDevice)
      : handle_base(), Platform(Platform), OffloadDevice(OffloadDevice) {}

  ur_platform_handle_t Platform;
  ol_device_handle_t OffloadDevice;

  // Profiling support. Liboffload reports elapsed time between two
  // profiling-enabled events on queues of the same device. We lazily create a
  // device-wide base event to act as a common origin (mirrors the CUDA
  // adapter's per-device EvBase) and report timestamps relative to it.
  std::once_flag BaseEventFlag;
  ol_queue_handle_t BaseQueue = nullptr;
  ol_event_handle_t BaseEvent = nullptr;

  // Return the timestamp (in nanoseconds) of the profiling-enabled event `Ev`
  // relative to this device's base event.
  ur_result_t getElapsedTime(ol_event_handle_t Ev, uint64_t &OutNs) {
    ur_result_t InitRes = UR_RESULT_SUCCESS;
    std::call_once(BaseEventFlag, [&]() {
      if (auto Err = olCreateQueue(OffloadDevice, &BaseQueue)) {
        InitRes = offloadResultToUR(Err);
        return;
      }
      if (auto Err = olCreateEvent(BaseQueue, OL_EVENT_FLAGS_ENABLE_PROFILING,
                                   &BaseEvent)) {
        InitRes = offloadResultToUR(Err);
        return;
      }
      if (auto Err = olSyncEvent(BaseEvent)) {
        InitRes = offloadResultToUR(Err);
      }
    });
    if (InitRes != UR_RESULT_SUCCESS) {
      return InitRes;
    }

    float Milliseconds = 0.0f;
    OL_RETURN_ON_ERR(olGetEventElapsedTime(BaseEvent, Ev, &Milliseconds));
    OutNs = static_cast<uint64_t>(Milliseconds * 1.0e6);
    return UR_RESULT_SUCCESS;
  }
};
