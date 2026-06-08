def segmented_scan [n] 't
                   (op: t -> t -> t)
                   (ne: t)
                   (flags: [n]bool)
                   (as: [n]t) : [n]t =
  (unzip (scan (\(x_flag, x) (y_flag, y) ->
                  ( x_flag || y_flag
                  , if y_flag then y else x `op` y
                  ))
               (false, ne)
               (zip flags as))).1

-- The old segmented reduce found in the segmented library.
def segmented_reduce [n] 't
                     (op: t -> t -> t)
                     (ne: t)
                     (flags: [n]bool)
                     (as: [n]t) =
  let as' = segmented_scan op ne flags as
  let segment_ends = rotate 1 flags
  let segment_end_offsets = segment_ends |> map i64.bool |> scan (+) 0
  let num_segments = if n > 0 then last segment_end_offsets else 0
  let scratch = replicate num_segments ne
  let index i f = if f then i - 1 else -1
  in scatter scratch (map2 index segment_end_offsets segment_ends) as'

def segscan_op op (v0, f0, i0) (v1, f1, i1) =
  (if f1 then v1 else v0 `op` v1, f0 || f1, i0 i64.+ i1)

def segreduce' [n] 't
               (op: t -> t -> t)
               (ne: t)
               (flags: [n]bool)
               (vals: [n]t) =
  let (segscans, _, offsets) =
    flags
    |> map i64.bool
    |> zip3 vals flags
    |> scan (segscan_op op) (ne, false, 0)
    |> unzip3
  let index f i = if f then i - 1 else -1
  let is = map2 index (rotate 1 flags) offsets
  let result = scatter (#[scratch] [vals][0]) is segscans
  let count =
    scatter (#[scratch] [0])
            (map (\j -> if j == n - 1 then 0 else -1) (iota n))
            offsets
  in result[:count[0]]

entry flags [n] (as: [n]i32) =
  map2 (\i a -> i == 0 || 0i32 < a) (iota n) as

-- ==
-- script input { ($loaddata "../data/randomints_full_500MiB.in", $loaddata "randomints_full_500MiB.flags") }
-- output @ randomints_full_500MiB.out
-- script input { ($loaddata "../data/randomints_dense_500MiB.in", $loaddata "randomints_dense_500MiB.flags") }
-- output @ randomints_dense_500MiB.out
-- script input { ($loaddata "../data/randomints_moderate_500MiB.in", $loaddata "randomints_moderate_500MiB.flags") }
-- output @ randomints_moderate_500MiB.out
-- script input { ($loaddata "../data/randomints_sparse_500MiB.in", $loaddata "randomints_sparse_500MiB.flags") }
-- output @ randomints_sparse_500MiB.out
-- script input { ($loaddata "../data/randomints_empty_500MiB.in", $loaddata "randomints_empty_500MiB.flags") }
-- output @ randomints_empty_500MiB.out
entry main [n] (as: [n]i32) (flags: [n]bool) =
  segreduce' (+) 0 flags as

-- ==
-- entry: expected
-- script input { ($loaddata "../data/randomints_full_500MiB.in", $loaddata "randomints_full_500MiB.flags") }
-- output @ randomints_full_500MiB.out
-- script input { ($loaddata "../data/randomints_dense_500MiB.in", $loaddata "randomints_dense_500MiB.flags") }
-- output @ randomints_dense_500MiB.out
-- script input { ($loaddata "../data/randomints_moderate_500MiB.in", $loaddata "randomints_moderate_500MiB.flags") }
-- output @ randomints_moderate_500MiB.out
-- script input { ($loaddata "../data/randomints_sparse_500MiB.in", $loaddata "randomints_sparse_500MiB.flags") }
-- output @ randomints_sparse_500MiB.out
-- script input { ($loaddata "../data/randomints_empty_500MiB.in", $loaddata "randomints_empty_500MiB.flags") }
-- output @ randomints_empty_500MiB.out
entry expected [n] (as: [n]i32) (flags: [n]bool) =
  segmented_reduce (+) 0 flags as
