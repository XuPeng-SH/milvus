// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once
#include <any>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>
#include <iostream>

namespace milvus::engine::snapshot {

using OnNoRefCBF = std::function<void(void)>;

class ReferenceProxy {
 public:
    ReferenceProxy() {
        ++CC;
        std::cout << "[CON] CC: " << CC << " DC: " << DC << std::endl;
    }
    virtual ~ReferenceProxy() {
        /* std::cout << "RC: " << ref_count_ << " CBS: " << on_no_ref_cbs_.size() << std::endl; */
        ++DC;
        std::cout << "[DES] CC: " << CC << " DC: " << DC << std::endl;
        /* on_no_ref_cbs_.clear(); */
    }
    /* virtual ~ReferenceProxy() = default; */

    // TODO: Copy constructor is used in Mock Test. Should never be used. To be removed
    ReferenceProxy(const ReferenceProxy& o) {
        ++CC;
        std::cout << "[CCON] CC: " << CC << " DC: " << DC << std::endl;
        ref_count_ = 0;
    }

    virtual void
    Ref() {
        ++ref_count_;
    }

    virtual void
    UnRef() {
        if (ref_count_ == 0) {
            return;
        }
        if (ref_count_.fetch_sub(1) == 1) {
            for (auto& cb : on_no_ref_cbs_) {
                cb();
            }
            on_no_ref_cbs_.clear();
        }
    }

    int64_t
    ref_count() const {
        return ref_count_;
    }

    void
    ResetCnt() {
        ref_count_ = 0;
    }

    void
    RegisterOnNoRefCB(const OnNoRefCBF& cb) {
        on_no_ref_cbs_.emplace_back(cb);
    }

 protected:
    std::atomic<int64_t> ref_count_ = {0};
    std::vector<OnNoRefCBF> on_no_ref_cbs_;
    static std::atomic_uint32_t CC;
    static std::atomic_uint32_t DC;
};

inline
std::atomic_uint32_t
ReferenceProxy::CC = 0;
inline
std::atomic_uint32_t
ReferenceProxy::DC = 0;

using ReferenceResourcePtr = std::shared_ptr<ReferenceProxy>;

}  // namespace milvus::engine::snapshot
