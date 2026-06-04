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
entry main [n] (as: [n]i32): [n]i32 =
  partition (0i32<) as
  |> uncurry (++)
  |> sized n

def partition_unordered [n] 'a
                        (p: a -> bool)
                        (as: [n]a) : [n]a =
  let to_index f (o0, o1) = if f then o0 - 1i64 else n - o1
  let add2 (a0, b0) (a1, b1) = (a0 + a1, b0 + b1)
  let t_flags = map p as
  let f_flags = map (\x -> !x) t_flags
  let flags =
    map2 (\x y ->
            ( i64.bool x
            , i64.bool y
            ))
          t_flags
          f_flags
  let offsets = scan add2 (0, 0) flags
  let idxs = map2 to_index t_flags offsets
  in scatter (#[scratch] [as][0]) idxs as

-- ==
-- entry: unordered
-- input @ ../data/randomints_full_500MiB.in
-- input @ ../data/randomints_dense_500MiB.in
-- input @ ../data/randomints_moderate_500MiB.in
-- input @ ../data/randomints_sparse_500MiB.in
-- input @ ../data/randomints_empty_500MiB.in
entry unordered [n] (as: [n]i32): [n]i32 =
  partition_unordered (0i32<) as