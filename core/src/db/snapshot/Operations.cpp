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

#include "db/snapshot/Operations.h"
#include <chrono>
#include "db/snapshot/OperationExecutor.h"
#include "db/snapshot/Snapshots.h"

namespace milvus {
namespace engine {
namespace snapshot {

static ID_TYPE UID = 1;

Operations::Operations(const OperationContext& context, ScopedSnapshotT prev_ss)
    : context_(context), prev_ss_(prev_ss), uid_(UID++), status_(40005, "Operation Pending") {
}

Operations::Operations(const OperationContext& context, ID_TYPE collection_id, ID_TYPE commit_id)
    : context_(context), uid_(UID++), status_(40005, "Operation Pending") {
    auto status = Snapshots::GetInstance().GetSnapshot(prev_ss_, collection_id, commit_id);
    if (!status.ok()) prev_ss_ = ScopedSnapshotT();
}

ID_TYPE
Operations::GetID() const {
    return uid_;
}

Status
Operations::operator()(Store& store) {
    return ApplyToStore(store);
}

void
Operations::SetStatus(const Status& status) {
    status_ = status;
}

Status
Operations::WaitToFinish() {
    std::unique_lock<std::mutex> lock(finish_mtx_);
    finish_cond_.wait(lock, [this] { return done_; });
    return status_;
}

void
Operations::Done() {
    std::unique_lock<std::mutex> lock(finish_mtx_);
    done_ = true;
    finish_cond_.notify_all();
}

Status
Operations::Push(bool sync) {
    return OperationExecutor::GetInstance().Submit(shared_from_this(), sync);
}

bool
Operations::IsStale() const {
    ScopedSnapshotT curr_ss;
    auto status = Snapshots::GetInstance().GetSnapshot(curr_ss, prev_ss_->GetCollectionId());
    if (!status.ok()) return true;
    if (prev_ss_->GetID() == curr_ss->GetID()) {
        return false;
    }

    return true;
}

Status
Operations::DoneRequired() const {
    Status status;
    if (!done_) {
        status = Status(40031, "Operation is expected to be done");
    }
    return status;
}

Status
Operations::IDSNotEmptyRequried() const {
    Status status;
    if (ids_.size() == 0)
        status = Status(40032, "No Snapshot is available");
    return status;
}

Status
Operations::PrevSnapshotRequried() const {
    Status status;
    if (!prev_ss_) {
        status = Status(40052, "Prev snapshot is requried");
    }
    return status;
}

Status
Operations::GetSnapshot(ScopedSnapshotT& ss) const {
    auto status = PrevSnapshotRequried();
    if (!status.ok()) return status;
    status = DoneRequired();
    if (!status.ok()) return status;
    status = IDSNotEmptyRequried();
    if (!status.ok()) return status;
    status = Snapshots::GetInstance().GetSnapshot(ss, prev_ss_->GetCollectionId(), ids_.back());
    return status;
}

Status
Operations::ApplyToStore(Store& store) {
    if (done_) return status_;
    auto status = OnExecute(store);
    SetStatus(status);
    Done();
    return status_;
}

Status
Operations::OnExecute(Store& store) {
    auto status = PreExecute(store);
    if (!status.ok()) {
        return status;
    }
    status = DoExecute(store);
    if (!status.ok()) {
        return status;
    }
    return PostExecute(store);
}

Status
Operations::PreExecute(Store& store) {
    return Status::OK();
}

Status
Operations::DoExecute(Store& store) {
    return Status::OK();
}

Status
Operations::PostExecute(Store& store) {
    return store.DoCommitOperation(*this);
}

}  // namespace snapshot
}  // namespace engine
}  // namespace milvus
