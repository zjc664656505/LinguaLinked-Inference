
/*!
 *  Copyright (c) 2023 by Contributors
 * \file huggingface_tokenizer.cc
 * \brief Huggingface tokenizer
 */
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

#include <cassert>

namespace tokenizers {
/*!
 * \brief A simple c++ header of tokenizer via C API.
 */
    class HFTokenizer : public Tokenizer {
    public:
        explicit HFTokenizer(TokenizerHandle handle) : handle_(handle) {}

        HFTokenizer(const HFTokenizer&) = delete;
        HFTokenizer(HFTokenizer&& other) { std::swap(other.handle_, handle_); }

        ~HFTokenizer() {
          if (handle_ != nullptr) {
            tokenizers_free(handle_);
          }
        }

        // use i32 to be consistent with sentencepiece
        std::vector<int32_t> Encode(const std::string& text) final {
          bool add_special_token = false;
          tokenizers_encode(handle_, text.data(), text.length(), static_cast<int>(add_special_token));
          const uint32_t* data;
          size_t len;
          tokenizers_get_encode_ids(handle_, &data, &len);
          const int32_t* data_i32 = reinterpret_cast<const int32_t*>(data);
          auto res = std::vector<int32_t>(data_i32, data_i32 + len);
          return res;
        }

        // use i32 to be consistent with sentencepiece
        std::string Decode(const std::vector<int32_t>& ids) final {
          bool skip_special_token = false;
          tokenizers_decode(handle_, reinterpret_cast<const uint32_t*>(ids.data()), ids.size(),
                            static_cast<int>(skip_special_token));
          const char* data;
          size_t len;
          tokenizers_get_decode_str(handle_, &data, &len);
          return std::string(data, len);
        }

        size_t GetVocabSize() final {
          size_t size;
          tokenizers_get_vocab_size(handle_, &size);
          assert(size > 0);
          return size;
        }

        std::string IdToToken(int32_t id) final {
          const char* data;
          size_t len;
          tokenizers_id_to_token(handle_, static_cast<uint32_t>(id), &data, &len);
          return std::string(data, len);
        }

        int32_t TokenToId(const std::string& token) final {
          int32_t id;
          tokenizers_token_to_id(handle_, token.data(), token.length(), &id);
          return id;
        }

    private:
        // internal handle
        TokenizerHandle handle_{nullptr};
    };

    std::unique_ptr<Tokenizer> Tokenizer::FromBlobJSON(const std::string& json) {
      return std::make_unique<HFTokenizer>(tokenizers_new_from_str(json.data(), json.length()));
    }

    std::unique_ptr<Tokenizer> Tokenizer::FromBlobByteLevelBPE(const std::string& vocab,
                                                               const std::string& merges,
                                                               const std::string& added_tokens) {
      return std::make_unique<HFTokenizer>(byte_level_bpe_tokenizers_new_from_str(
              vocab.data(), vocab.length(), merges.data(), merges.length(), added_tokens.data(),
              added_tokens.length()));
    }
}  // namespace tokenizers
