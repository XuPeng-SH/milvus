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

#include "db/SnapshotVisitor.h"
#include <sstream>
#include "db/SnapshotHandlers.h"
#include "db/meta/MetaTypes.h"
#include "db/snapshot/Snapshots.h"

namespace milvus {
namespace engine {

SnapshotVisitor::SnapshotVisitor(snapshot::ScopedSnapshotT ss) : ss_(ss) {
}

SnapshotVisitor::SnapshotVisitor(const std::string& collection_name) {
    status_ = snapshot::Snapshots::GetInstance().GetSnapshot(ss_, collection_name);
}

SnapshotVisitor::SnapshotVisitor(snapshot::ID_TYPE collection_id) {
    status_ = snapshot::Snapshots::GetInstance().GetSnapshot(ss_, collection_id);
}

Status
SnapshotVisitor::SegmentsToSearch(meta::FilesHolder& files_holder) {
    STATUS_CHECK(status_);

    auto handler = std::make_shared<SegmentsToSearchCollector>(ss_, files_holder);
    handler->Iterate();

    return handler->GetStatus();
}

FieldElementVisitor::Ptr
FieldElementVisitor::Build(snapshot::ScopedSnapshotT ss, snapshot::ID_TYPE segment_id,
                           snapshot::ID_TYPE field_element_id) {
    if (!ss) {
        return nullptr;
    }

    auto element = ss->GetResource<snapshot::FieldElement>(field_element_id);
    if (!element) {
        return nullptr;
    }

    auto visitor = std::make_shared<FieldElementVisitor>();
    visitor->SetFieldElement(element);
    auto segment = ss->GetResource<snapshot::Segment>(segment_id);
    if (!segment) {
        return nullptr;
    }

    auto file = ss->GetSegmentFile(segment_id, field_element_id);
    if (!file) {
        return nullptr;
    }

    visitor->SetFile(file);
    return visitor;
}

SegmentFieldVisitor::Ptr
SegmentFieldVisitor::Build(snapshot::ScopedSnapshotT ss, snapshot::ID_TYPE segment_id, snapshot::ID_TYPE field_id) {
    if (!ss) {
        return nullptr;
    }

    auto field = ss->GetResource<snapshot::Field>(field_id);
    if (!field) {
        return nullptr;
    }

    auto visitor = std::make_shared<SegmentFieldVisitor>();
    visitor->SetField(field);

    auto& field_elements = ss->GetResources<snapshot::FieldElement>();
    for (auto& kv : field_elements) {
        if (kv.second->GetFieldId() != field_id) {
            continue;
        }
        auto element_visitor = FieldElementVisitor::Build(ss, segment_id, kv.first);
        if (!element_visitor) {
            continue;
        }
        visitor->InsertElement(element_visitor);
    }

    return visitor;
}

SegmentVisitor::Ptr
SegmentVisitor::Build(snapshot::ScopedSnapshotT ss, snapshot::ID_TYPE segment_id) {
    if (!ss) {
        return nullptr;
    }
    auto segment = ss->GetResource<snapshot::Segment>(segment_id);
    if (!segment) {
        return nullptr;
    }

    auto visitor = std::make_shared<SegmentVisitor>();
    visitor->SetSegment(segment);

    auto& fields = ss->GetResources<snapshot::Field>();
    for (auto& kv : fields) {
        auto field_visitor = SegmentFieldVisitor::Build(ss, segment_id, kv.first);
        if (!field_visitor) {
            continue;
        }
        visitor->InsertField(field_visitor);
    }

    return visitor;
}

std::string
SegmentVisitor::ToString() const {
    std::stringstream ss;
    ss << "SegmentVisitor[" << GetSegment()->GetID() << "]: \n";
    auto& field_visitors = GetFieldVisitors();
    for (auto& fkv : field_visitors) {
        ss << "  Field[" << fkv.first << "]\n";
        auto& fe_visitors = fkv.second->GetElementVistors();
        for (auto& fekv : fe_visitors) {
            ss << "    FieldElement[" << fekv.first << "] ";
            ss << "SegmentFile [" << fekv.second->GetFile()->GetID() << "]\n";
        }
    }

    return ss.str();
}

}  // namespace engine
}  // namespace milvus
