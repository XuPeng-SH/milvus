#ifndef VECENGINE_DB_META_IMPL_H_
#define VECENGINE_DB_META_IMPL_H_

#include "Meta.h"
#include "Options.h"

namespace zilliz {
namespace vecwise {
namespace engine {
namespace meta {

class DBMetaImpl : public Meta {
public:
    DBMetaImpl(const DBMetaOptions& options_);

    virtual Status add_group(const GroupOptions& options_,
            const std::string& group_id_,
            GroupSchema& group_info_) override;
    virtual Status get_group(const std::string& group_id_, GroupSchema& group_info_) override;
    virtual Status has_group(const std::string& group_id_, bool& has_or_not_) override;

    virtual Status add_group_file(const std::string& group_id,
                                  DateT date,
                                  GroupFileSchema& group_file_info,
                                  GroupFileSchema::FILE_TYPE file_type=GroupFileSchema::RAW) override;

    virtual Status add_group_file(const std::string& group_id_,
                                  GroupFileSchema& group_file_info_,
                                  GroupFileSchema::FILE_TYPE file_type=GroupFileSchema::RAW) override;

    virtual Status has_group_file(const std::string& group_id_,
                                  const std::string& file_id_,
                                  bool& has_or_not_) override;
    virtual Status get_group_file(const std::string& group_id_,
                                  const std::string& file_id_,
                                  GroupFileSchema& group_file_info_) override;
    virtual Status update_group_file(const GroupFileSchema& group_file_) override;

    virtual Status get_group_files(const std::string& group_id_,
                                   const int date_delta_,
                                   GroupFilesSchema& group_files_info_) override;

    virtual Status update_files(const GroupFilesSchema& files) override;

    virtual Status files_to_merge(const std::string& group_id,
            DatePartionedGroupFilesSchema& files) override;

    virtual Status files_to_index(GroupFilesSchema&) override;

private:

    Status initialize();

    const DBMetaOptions _options;

}; // DBMetaImpl

} // namespace meta
} // namespace engine
} // namespace vecwise
} // namespace zilliz

#endif // VECENGINE_DB_META_IMPL_H_
