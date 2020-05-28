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

#include <assert.h>
#include <any>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "Context.h"
#include "db/snapshot/Snapshot.h"
#include "db/snapshot/Store.h"
#include "utils/Status.h"

namespace milvus {
namespace engine {
namespace snapshot {

using StepsT = std::vector<std::any>;

/* enum OpStatus { */
/*     OP_PENDING = 0, */
/*     OP_OK, */
/*     OP_STALE_OK, */
/*     OP_STALE_CANCEL, */
/*     OP_STALE_RESCHEDULE, */
/*     OP_FAIL_INVALID_PARAMS, */
/*     OP_FAIL_DUPLICATED, */
/*     OP_FAIL_FLUSH_META */
/* }; */

class Operations : public std::enable_shared_from_this<Operations> {
 public:
    /* static constexpr const char* Name = Derived::Name; */
    Operations(const OperationContext& context, ScopedSnapshotT prev_ss);
    Operations(const OperationContext& context, ID_TYPE collection_id, ID_TYPE commit_id = 0);

    const ScopedSnapshotT&
    GetPrevSnapshot() const {
        return prev_ss_;
    }

    virtual bool
    IsStale() const;

    template <typename StepT>
    void
    AddStep(const StepT& step, bool activate = true);
    void
    SetStepResult(ID_TYPE id) {
        ids_.push_back(id);
    }

    StepsT&
    GetSteps() {
        return steps_;
    }

    ID_TYPE
    GetID() const;

    virtual Status
    OnExecute(Store&);
    virtual Status
    PreExecute(Store&);
    virtual Status
    DoExecute(Store&);
    virtual Status
    PostExecute(Store&);

    virtual ScopedSnapshotT
    GetSnapshot() const;

    virtual Status
    operator()(Store& store);
    virtual Status
    Push(bool sync = true);

    virtual Status
    ApplyToStore(Store& store);

    Status
    WaitToFinish();

    void
    Done();

    void SetStatus(const Status& status);

    Status GetStatus() const { return status_; }

    virtual ~Operations() {
    }

 protected:
    OperationContext context_;
    ScopedSnapshotT prev_ss_;
    StepsT steps_;
    std::vector<ID_TYPE> ids_;
    bool done_ = false;
    Status status_;
    mutable std::mutex finish_mtx_;
    std::condition_variable finish_cond_;
    ID_TYPE uid_;
};

template <typename StepT>
void
Operations::AddStep(const StepT& step, bool activate) {
    auto s = std::make_shared<StepT>(step);
    if (activate)
        s->Activate();
    steps_.push_back(s);
}

template <typename ResourceT>
class CommitOperation : public Operations {
 public:
    using BaseT = Operations;
    CommitOperation(const OperationContext& context, ScopedSnapshotT prev_ss) : BaseT(context, prev_ss) {
    }
    CommitOperation(const OperationContext& context, ID_TYPE collection_id, ID_TYPE commit_id = 0)
        : BaseT(context, collection_id, commit_id) {
    }

    virtual typename ResourceT::Ptr
    GetPrevResource() const {
        return nullptr;
    }

    typename ResourceT::Ptr
    GetResource(bool wait = false) {
        if (wait) {
            WaitToFinish();
        }
        if (ids_.size() == 0)
            return nullptr;
        resource_->SetID(ids_[0]);
        return resource_;
    }

    // PXU TODO
    /* virtual void OnFailed() */

 protected:
    typename ResourceT::Ptr resource_;
};

template <typename ResourceT>
class LoadOperation : public Operations {
 public:
    explicit LoadOperation(const LoadOperationContext& context)
        : Operations(OperationContext(), ScopedSnapshotT()), context_(context) {
    }

    Status
    ApplyToStore(Store& store) override {
        if (done_) return status_;
        resource_ = store.GetResource<ResourceT>(context_.id);
        SetStatus(Status::OK());
        Done();
        return status_;
    }

    typename ResourceT::Ptr
    GetResource(bool wait = false) {
        if (wait) {
            WaitToFinish();
        }
        if (!done_) return nullptr;
        return resource_;
    }

 protected:
    LoadOperationContext context_;
    typename ResourceT::Ptr resource_;
};

template <typename ResourceT>
class HardDeleteOperation : public Operations {
 public:
    explicit HardDeleteOperation(ID_TYPE id) : Operations(OperationContext(), ScopedSnapshotT()), id_(id) {
    }

    Status
    ApplyToStore(Store& store) override {
        if (done_) return status_;
        auto status = store.RemoveResource<ResourceT>(id_);
        SetStatus(status);
        Done();
        return status_;
    }

 protected:
    ID_TYPE id_;
};

using OperationsPtr = std::shared_ptr<Operations>;

}  // namespace snapshot
}  // namespace engine
}  // namespace milvus
