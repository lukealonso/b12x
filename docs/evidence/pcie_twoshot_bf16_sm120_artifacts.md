# BF16 PCIe two-shot artifact map

Current implementation status: **qualified** for B12X commit
`bf3699b87fc1751e4eccf10f18361799d5ef8b86`. A fresh isolated compile cache
contained exactly 36 manifests and 36 objects covering three operations, four
ranks, and eager plus both graph-slot modes. Every object matched the SHA-256
stored in its manifest, every cache key matched its manifest path, and every
manifest carried package fingerprint
`7171ff95b5efdb9bb27787ef64e788866560c83394cbf665db4df38e7afedbfd`.
The qualification command and exact source trees are recorded in
[`pcie_twoshot_bf16_sm120.md`](pcie_twoshot_bf16_sm120.md).

The indexed table below is **qualified historical test evidence** for B12X
commit `7edd604a621ddbc3db1545e54d0e7031090bace5`; its cache keys and hashes are
not attributed to the qualified implementation commit above.

The isolated compile cache contained 36 CUTLASS DSL objects: three collective
operations, four ranks, and three slot-selection modes per rank. `eager` means
host-selected double buffering; `graph-0` and `graph-1` mean device-selected
double buffering with the stated synchronized slot bias. Every manifest binds
Python 3.12.3, PyTorch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2,
cuda-bindings 13.3.1, PTXAS 13.3.73, 512 threads, and 4,096 BF16 elements per
row. The package fingerprint recorded by every manifest is
`c5ec72e5f0f926a5969d3613bbab2fad26c61959d985a259db4c3258a3cfbfc9`.

## Provenance and verification

The artifacts were generated from B12X commit
`7edd604a621ddbc3db1545e54d0e7031090bace5`, whose repository tree is
`19f23a8eeb2dc5f6eadceee791afae9a545f2eaf` and whose `b12x/` package tree is
`5c13b2d9809025c5bf83c9ddb9071352acb60c0f`. The package fingerprint is the
SHA-256 digest produced by visiting every regular file below `b12x/` in sorted
relative-path order, excluding `__pycache__`, `.pyc`, and `.pyo` files, and
hashing `relative path + NUL + file bytes + NUL` for each file.

Each cache key is `SHA256(repr(cache_payload).encode("utf-8"))`; the exact
`cache_payload` and its representation are stored in the corresponding JSON
manifest. Given a compile-cache directory named `CACHE_DIR`, a row's durable
relative locators are
`$CACHE_DIR/${cache_key:0:2}/$cache_key.json` for the manifest and
`$CACHE_DIR/${cache_key:0:2}/$cache_key.o` for the object. “Manifest SHA-256”
and “Object SHA-256” are hashes of the complete raw file bytes at those
locations. This versioned table is the manifest index; generated object files
are build outputs and are not stored in Git.

The qualification run mounted its empty host cache at `/test-cache/cute` in
the test container. `B12X_COMPILE_CACHE_DIR=/test-cache/cute` selected that
directory for compiler objects, while
`B12X_CUTE_COMPILE_CACHE_DIR=/test-cache/cute` recorded the same durable
location in each manifest's compile environment. After running the four-rank
correctness command from the qualification document, the following verifier
checks the exact source identity plus all 36 indexed manifests and objects.
`CACHE_DIR` must name the resulting compile-cache directory,
`B12X_SOURCE_ROOT` must name a checkout of the source revision above, and
`INDEX_PATH` must name this reviewed artifact-map file. The artifact map need
not exist in the historical source checkout.

```bash
readonly CACHE_DIR=/test-cache/cute
readonly B12X_SOURCE_ROOT=/path/to/b12x-at-7edd604a621ddbc3db1545e54d0e7031090bace5
readonly INDEX_PATH="$(git rev-parse --show-toplevel)/docs/evidence/pcie_twoshot_bf16_sm120_artifacts.md"
readonly EXPECTED_COMMIT=7edd604a621ddbc3db1545e54d0e7031090bace5
readonly EXPECTED_REPOSITORY_TREE=19f23a8eeb2dc5f6eadceee791afae9a545f2eaf
readonly EXPECTED_B12X_TREE=5c13b2d9809025c5bf83c9ddb9071352acb60c0f
test "$(git -C "$B12X_SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git -C "$B12X_SOURCE_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_REPOSITORY_TREE"
test "$(git -C "$B12X_SOURCE_ROOT" rev-parse HEAD:b12x)" = "$EXPECTED_B12X_TREE"
test -f "$INDEX_PATH"
python - "$CACHE_DIR" "$B12X_SOURCE_ROOT" "$INDEX_PATH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

cache_dir, source_root, index_path = map(Path, sys.argv[1:])

package_digest = hashlib.sha256()
package_root = source_root / "b12x"
package_files = sorted(
    path
    for path in package_root.rglob("*")
    if path.is_file()
    and "__pycache__" not in path.parts
    and path.suffix not in {".pyc", ".pyo"}
)
for path in package_files:
    package_digest.update(str(path.relative_to(package_root)).encode("utf-8"))
    package_digest.update(b"\0")
    package_digest.update(path.read_bytes())
    package_digest.update(b"\0")
expected_package = "c5ec72e5f0f926a5969d3613bbab2fad26c61959d985a259db4c3258a3cfbfc9"
assert package_digest.hexdigest() == expected_package

rows = 0
for line in index_path.read_text().splitlines():
    if not line.startswith("| ") or line.startswith("| Operation"):
        continue
    cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
    if len(cells) != 7:
        continue
    cache_key, manifest_sha, object_sha = cells[-3:]
    artifact_dir = cache_dir / cache_key[:2]
    manifest_path = artifact_dir / f"{cache_key}.json"
    object_path = artifact_dir / f"{cache_key}.o"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_sha
    assert hashlib.sha256(object_path.read_bytes()).hexdigest() == object_sha
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cache_key"] == cache_key
    assert manifest["object_sha256"] == object_sha
    assert manifest["package_fingerprint"] == expected_package
    rows += 1
assert rows == 36, rows
print("verified 36 BF16 PCIe two-shot manifests and objects")
PY
```

| Operation | Rank | Physical GPU UUID | Slot mode | Cache key | Manifest SHA-256 | Object SHA-256 |
|:--|--:|:--|:--|:--|:--|:--|
| all-gather | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | eager | `a322403cd930142c5376f33b27cd86a430a90ed2cd5311ec14e64b91c1c6b86b` | `4734c61814755f8a85f3be2d49426a6abc9b92d5082e3b6ad9be243d41cc9f1d` | `e6645414192f28d00fa1256017a5e6277151aac8d25fdb618b1509afa5a5df70` |
| all-gather | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-0 | `10d5b18a40281325d7c9e260bb7ec9bc6ca0b625c8f640c66f19869e528598b4` | `6f001a44ab34f5d16bc265277649eb38f0711622b87baf28b798547301f144ca` | `e336ac851824b02d06c0caf2e178c378f86ee904326054a9522e23dc23db1ac7` |
| all-gather | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-1 | `a5b8c2188db473e102147fad72b9df7db711ccce701666f84c791a5bb9307e1e` | `754ae31c0afc14052d8ca6b5b96aff6530461f4de3f8f4296db2efe8c8d1ca0a` | `cf49e134b3f541041b9a064fceb118a01d7ffeef858543edae27092f4f069770` |
| all-gather | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | eager | `04722d1397c4309f08eb5252ca1c8ef9dcf6f1f6613303720641d2814248e970` | `c9058e9dea5293454cd7a4c00053c3570fb257e1dc1364391b3c9510b75f513a` | `56429c632aa8f30ca5322e8cccd02e7d6de5b38fb59e5ce827442635807b6fc2` |
| all-gather | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-0 | `c2c66c9561b323494674add678e4f69c83a7285fb2fc568987fb7f78df03abc8` | `4afd6a8960543bcd27d884275afb2de6334459ab7a654ec0a948ec226d98dcfd` | `169874859e6d8687ae2f8d008999abf3d6af6d39d880cb5a7bc13cb744b14dc1` |
| all-gather | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-1 | `4110c854e35d994fa120176545d0f7b2d344ce923121aff82ac266034f896d6a` | `7d792b555fbc396e515af375c45676867bf675c15c9d6bd3abfdebeea26da927` | `593d13b5e1ab033dab95f28bf1b1df5b34a6decbf4121c60bc33f98c473c442a` |
| all-gather | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | eager | `4dfe8fe97273dd2f95fb7d2da7f7cb3d7ff4a7f59cc7ea9e6bd3d32f58cf95da` | `f077d6fc71243c4eaa9c70ad7f3662a85d8b635c66d20ae31bb801b4156252f0` | `c63a207f9c9a13b1150daa8b4ba294f01f0eb98a71efa8665128538ef40d32fc` |
| all-gather | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-0 | `2688eaafdcccda2d5d33b693ad0547bb642f7804fd1723b7ab31f634541fde74` | `5ea82b6b42fac3a0457c6f0fbc82c55473149e9f69b6d8872f3ddffcee1aecba` | `073c71c879b830b85ea6350cd1bb84b0ac816092459292ee35be650b85d3d1a1` |
| all-gather | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-1 | `a3ec6bcc4de3b6ec79e1ebc3cd8e68103170d80336aae114fe5de820e5da3e18` | `9bb2401d63017a14bac8e83191cec15351bdf406a7e1cd07fdd591e247c8c646` | `b3526feb44aa4325790fee9fca97973fc531bf7aa5c5b721269f0f00d5f19728` |
| all-gather | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | eager | `16a30fc2f03e75f395c3502ce11aab4166c6b7d063982ae9971a0e6243b91a72` | `555de19b3e7a3807e3db90f1e8f97966d3dd7b9d79c51c6e6a20cf07106f69bb` | `a57dc7256c555b09ce7648f02f570614f1da31629f1550240485eb3011fc4728` |
| all-gather | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-0 | `f7b457aa303cc1a72b42f2c39a8133ef3a829db3261b43961c0e60d33f996018` | `dbec545852e43607a3964694df38a4c9a7abc6f9d3b60753ceb3b5236e0d46cd` | `7ce91796b8fcd75e15403c34a3999ae0aceec0109e91751d070813af556c3442` |
| all-gather | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-1 | `be350c871e5efbecc108ee99fae3ccb20713208d25b185ba958b900f4e19a829` | `4fe3f8a2cbde40a31ed3364d1bb05110a3897eadd93f95f8e5ca158d4402ca1c` | `2f3d46c02e10cade0c2dda91ded3cf333530e940bdd291dbb001c6346e3e9d9f` |
| pull all-reduce | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | eager | `aad96e32d74bb6406a730f3cc8172f7a6795386b89b79d7c25857cb9da51114b` | `db312d4c5472009fd8f85b76ba614250a8f57bd3128655abd3fae43f702f9805` | `6cb9cc0c200dabbcef5e552f883049f5f38564177dae94ee56a33f45af436a8d` |
| pull all-reduce | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-0 | `2e6b88f15782a001007891243dce62d59002dbf05f343747a5bb556b8851b6aa` | `89a19dee0265e4662b01c44858b97b4b3ae2dc42d5f3cc07e456e08848fc555d` | `f06047ab724e12f124b1bca95ff3078eb963cecd4efd4ca8955334159623da63` |
| pull all-reduce | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-1 | `4f55f273ad21220ebb0bd1cd03480c4ee2c10608acbd91623e4cef95fc502b30` | `437e8928c908020eff768176865eab20435fe1e0eedfe91e6da5ffd23bdc5702` | `d1ab7f3eb33de49a4a27ba84a360bff68e3204a3030179c5c79e7c8626e5dded` |
| pull all-reduce | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | eager | `5ab6f7504d3fd38136241081d8227de587c793bec3b7e00beca27089df787e1a` | `d1013474bf157fed67fcdcda7780875d9bdcc2fe5bb08131f4dbe67da9827d94` | `3e0e0db2add6e6fc9d654c94f6fda5deb6e6d40192ef62ebe64ad4f949abd864` |
| pull all-reduce | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-0 | `da7ac3e95f3c8ea21d1e507036dcea927ef009990756b1c17d3669e135cbe3d6` | `e9e6cf7281a82efedfa7f2f616437530966b76ae00aed521515559fd42b454e1` | `d5a208e2579fcf65b5d4bbe11fb717813e7222896efe8aad186744577fe267eb` |
| pull all-reduce | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-1 | `0e3fbca47753dd44cf11d8b797f180fd953ffa025bb5ef9ed8bbee1987c8a7d2` | `827dc51ef5bf71dd27bed6dc88871834ce50929f0de381e6e224b8e3cc7bb106` | `eab4a8dcdea80ff7f5c6132adafd606a55893413c27cbfc470b75779cc4b9bd7` |
| pull all-reduce | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | eager | `400031c0f2302a1c034ef40eeec89cf65662acb53a499ad1ede4ffc3e3675560` | `86d66b1b8f25875b63bc67250bc811865ca76c903ba8a762d10526c269d350ee` | `5d6d9b5cdbbb27ec39b0931ce3b09ef9167fd7b3b6f06ba3d163b8f3ba2beba7` |
| pull all-reduce | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-0 | `58c3a7f56c2d0c611a0c08ec0cd9044e984bb77e08dd534a94d153f9e4e499ab` | `a1b341874b8b3d6bc8e128d6d216e3e0101795fc1bc556f6d39f254bcf667be3` | `3d017ea34e3fad6c87a31dc293846eee6adf9db52a13daf094cef01dfad47707` |
| pull all-reduce | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-1 | `709f852ddab021ca314db659616a60759dbd3571959568a014de15b51cbb48a2` | `96132a1c384c184f48af7ae6b6efeb6244ce253c6901c8d32c035939532243e8` | `cfd6b2530dc33b302919f38ce17c2b14739c3a0db0d40a8a5d6fb2c51e7b7825` |
| pull all-reduce | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | eager | `b7c3b6aa699fcb4544af6cf168c7180ce91e58845303ab44d382460289ecccba` | `c4a761c4f8c596fa984fe6d3b676f04399565d4242a1a8a2ccee5852216e7d63` | `e1d0cff3a15e6f282409e7ba8ff3cb4a67266923d43163323f6f008631e918a9` |
| pull all-reduce | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-0 | `8d6b62685b7db3faa02dae195252fb65c25b4aad53c862f3d7e479549454618d` | `6189b08c7f1da269b0b5c45f2349cb33f078d9a0960fd4d5699d3d8491b3964a` | `c80d1cfbc34f6d2c67060f5f009bceb9bd341c206a3ae9b52269592ba2620c50` |
| pull all-reduce | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-1 | `d086c4bdbdd3f22393b1520b28a212806a2344a237d8302099376ca89c19bd79` | `2470441544bc233fbad4f59d632c28d550738fd7e52da228f0f91af0273269a4` | `e0d8a419c5552f26ddc47cd5cb8b1a420ad847629e1c08aa48495b600e8b17c4` |
| reduce-scatter | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | eager | `ce80d2d0f65da404b4b6b2b94e7cad7cdfbe0a8aa7f3cb7c99d463a3af11775b` | `6bf93b85bade08beed3151c9d4919e4c7fcf1ff48e40c64ec7a7b685f2911e9c` | `328bdf984acdddf11215fb4178f22820c6113718d50faa815688bd7cd6f95ac0` |
| reduce-scatter | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-0 | `7b0553ca85937136dc0ce912c27e5b50c8204ffdb3858c825ff2c86b6cf40b93` | `85b7102d84337af832be469dd63aeadecd7b30b9bb1bce7dafd172c1dcba2fb6` | `008a19b6d03f1e9d154f1108d222f6838e47a67f03cd53d989a243fb37abdb3e` |
| reduce-scatter | 0 | `8800cf0c-1ba5-7136-d796-2a91f9e9586e` | graph-1 | `fab3d561b0d05f4afdc7805d3374b3b8cd5b8c4c7ce902947c39b6c351704f53` | `e67e8b2d16712af18ed02a3d1d6d3c08ff0c37f089e549453c174dfb1eb6c4a1` | `d70bc8d4bd1c8fcecab11fdcd61efb7003cd8f7caf2cb8b4529fa1b3ec6d1acf` |
| reduce-scatter | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | eager | `a297f2c8192c60074f051ceb892ef9b75bfe10601b152875ecc4d52f0d284f55` | `bb79625fec2b87c027097a5f61b25e282a3eaa266b27a79f7c491bf56aaa6a08` | `d261ee7d8dd0007299e19829e051131fecc1b40b66343660cde5be064f11e7a8` |
| reduce-scatter | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-0 | `14ad78b7256850e901ae1506ff4290e3d1f3ef98420091ce6a21580b02d6c450` | `ee77f5bf1b5a12345c17d3c5bd73b0e2d88abe81e606c4b281609e20083e01e5` | `2c9531aad6d5e37f31cbf4acb5ee6f8faa4742c787eb8227225260aa269b367a` |
| reduce-scatter | 1 | `4a0aa20b-8e36-2e05-4efb-8befbf1181d4` | graph-1 | `71ebef67125774d158dbebe21b381ea8f40027e320f398a37d8a8157c7e3edf5` | `1522189018183f39c333e2364c7fa2c17bf22f3e250ca76f554531059ae40ee0` | `ff4bc8b4c491e0400772670a2231098d53b2880cf6dc4caa7c41c8e8f8113f2a` |
| reduce-scatter | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | eager | `f8585470429d621c490b0e60506a0a73a2cd727e23240e708e77045d61eea7cf` | `d9240cbd39c7247cecfe1eaf3647b8e13168aa726b1dd13e8958e0b85f3f4489` | `eca681e65bd8275f8452cd16d4da90185aa4c91478022cfd47e58203adc95c60` |
| reduce-scatter | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-0 | `4592686c4f64c09419decfdfde5b9a4044cbe1001909e9f23d99b80410c27a38` | `9d5bbcfbb021d8535bb0e4ddb99072768d0184a72f7d21c55f025ae2e731aac5` | `d498b0d9b0f39940672e2c6ec23f0e090d0016c8fc2f0b2117dc58e173cea845` |
| reduce-scatter | 2 | `1a0323f7-8113-a1e1-c68b-f23fecf77171` | graph-1 | `af9f51d3a0acb5ef0f3885c7e7ce29d0a2296bcdfe746844fe1d1c7b4c2c6c93` | `cfcbc00589970200960674289d9ffdf282eb452ca80e5afe6aca8687b7df283b` | `edc21e08659400be5db16cf4ac6eb34c3265814642f0085f5331738c4228c85b` |
| reduce-scatter | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | eager | `68cbd638666b40cc5344148f4bd7098658039801b77e0cfef63ddd96d8307a23` | `df0a1956bc4e19cf8c54e664a4e1d8e0674c9410c707d161b694260f37eaed8d` | `827c558f43f57f2d3918b93d623e9eaa38e743454e169c3268e3f46baa6cc5bd` |
| reduce-scatter | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-0 | `b8e40d5e6a91ad6668d14920e8b7fe08e57b335c0fece30cee0fe487de324780` | `8238b9568cd5598558793c7ecc11d843c7dc9e70832b91d6e6b18f700421000b` | `6d166205c0992dd4d5fd9d3d4805d8a8c2f9e92c4fd321d80e6f776f180afa83` |
| reduce-scatter | 3 | `0027fc86-3322-ce2a-856c-f49eb61eb63e` | graph-1 | `db1897654470cb1f9c9f18ad233686b7cbb1351801fbffa409a9c3fd462c5b88` | `5d324d9cc0fdd58d7e7e07b33ab45d003b2e39463d636b0aaf3b0e63bdb77dc6` | `557d478cea3d365a274d62eacba4ddf788ea38d909c871feb0b39183535ffc6c` |
