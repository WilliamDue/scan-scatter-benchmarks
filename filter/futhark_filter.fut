def expected_filter [n] 'a (p: a -> bool) (as: [n]a) : *[]a =
  let flags = map (\x -> if p x then 1i32 else 0i32) as
  let offsets = scan (+) 0i32 flags
  let m = if n == 0 then 0 else offsets[n - 1]
  in scatter (#[scratch] map (\x -> x) as[:i64.i32 m])
             (map2 (\f o -> i64.i32 (if f == 1 then o - 1 else -1)) flags offsets)
             as

def filter' [n] 'a (p: a -> bool) (as: [n]a) : *[]a =
  let flags = map p as
  let offsets = scan (+) 0 (map i32.bool flags)
  let flags' = map p as
  let is = map2 (\f o -> if f then o - 1i32 else -1i32) flags' offsets
  -- This following is carefully written such that the two scatters will be
  -- fused horisontally, which allows the entire thing to become a single
  -- kernel.
  let result = scatter (#[scratch] [as][0]) (map i64.i32 is) as
  let count =
    scatter [0]
            (map (\j -> if j == n - 1 then 0 else -1) (0..1..<n))
            offsets
  in result[:i64.i32 count[0]]

-- ==
-- entry: main
-- input @ ../data/randomints_full_500MiB.in
-- output @ randomints_full_500MiB.out
-- input @ ../data/randomints_dense_500MiB.in
-- output @ randomints_dense_500MiB.out
-- input @ ../data/randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
-- input @ ../data/randomints_sparse_500MiB.in
-- output @ randomints_sparse_500MiB.out
-- input @ ../data/randomints_empty_500MiB.in
-- output @ randomints_empty_500MiB.out
entry main [n] (as: [n]i32) : *[]i32 =
  filter' (0i32 <) as

-- ==
-- entry: expected
-- input @ ../data/randomints_full_500MiB.in
-- output @ randomints_full_500MiB.out
-- input @ ../data/randomints_dense_500MiB.in
-- output @ randomints_dense_500MiB.out
-- input @ ../data/randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
-- input @ ../data/randomints_sparse_500MiB.in
-- output @ randomints_sparse_500MiB.out
-- input @ ../data/randomints_empty_500MiB.in
-- output @ randomints_empty_500MiB.out
entry expected [n] (as: [n]i32) : *[]i32 =
  expected_filter (0i32 <) as
