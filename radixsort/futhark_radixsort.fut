def get_bit_i32 i x =
  let b = i32.get_bit i x
  in if i == i32.num_bits - 1 then b ^ 1 else b

local
def old_radix_sort_step [n] 't
                        (xs: [n]t)
                        (get_bit: i32 -> t -> i32)
                        (digit_n: i32) : [n]t =
  let num x = get_bit (digit_n + 1) x * 2 + get_bit digit_n x
  let pairwise op (a1, b1, c1, d1) (a2, b2, c2, d2) = (a1 `op` a2, b1 `op` b2, c1 `op` c2, d1 `op` d2)
  let bins = xs |> map num
  let flags =
    bins
    |> map (\x ->
              ( i64.bool (x == 0)
              , i64.bool (x == 1)
              , i64.bool (x == 2)
              , i64.bool (x == 3)
              ))
  let offsets = scan (pairwise (+)) (0, 0, 0, 0) flags
  let (na, nb, nc, _nd) = last offsets
  let f bin (a, b, c, d) =
    (-1)
    + a * (i64.bool (bin == 0))
    + na * (i64.bool (bin > 0))
    + b * (i64.bool (bin == 1))
    + nb * (i64.bool (bin > 1))
    + c * (i64.bool (bin == 2))
    + nc * (i64.bool (bin > 2))
    + d * (i64.bool (bin == 3))
  let is = map2 f bins offsets
  in scatter (copy xs) is xs

-- | Old Radix Sort
def old_radix_sort [n] 't
                   (num_bits: i32)
                   (get_bit: i32 -> t -> i32)
                   (xs: [n]t) : [n]t =
  let iters = if n == 0 then 0 else (num_bits + 2 - 1) / 2
  in loop xs for i < iters do old_radix_sort_step xs get_bit (i * 2)

local
def old_with_fuse_radix_sort_step [n] 't
                                  (xs: [n]t)
                                  (get_bit: i32 -> t -> i32)
                                  (digit_n: i32) : [n]t =
  let num x = i8.i32 <| get_bit (digit_n + 1) x * 2 + get_bit digit_n x
  let pairwise op (a1, b1, c1, d1) (a2, b2, c2, d2) = (a1 `op` a2, b1 `op` b2, c1 `op` c2, d1 `op` d2)
  let pairwise' op (a1, b1, c1) (a2, b2, c2) = (a1 `op` a2, b1 `op` b2, c1 `op` c2)
  let bins = map num xs
  let flags =
    bins
    |> map (\x -> (x == 0, x == 1, x == 2, x == 3))
  let flags' =
    map num xs
    |> map (\x ->
              ( i64.bool (x == 0)
              , i64.bool (x == 1)
              , i64.bool (x == 2)
              ))
  let offsets =
    flags
    |> map (\(a, b, c, d) -> (i64.bool a, i64.bool b, i64.bool c, i64.bool d))
    |> scan (pairwise (+)) (0, 0, 0, 0)
  let (na, nb, nc) = reduce_comm (pairwise' (+)) (0, 0, 0) flags'
  let f bin (a, b, c, d) =
    (-1)
    + a * (i64.bool (bin == 0))
    + na * (i64.bool (bin > 0))
    + b * (i64.bool (bin == 1))
    + nb * (i64.bool (bin > 1))
    + c * (i64.bool (bin == 2))
    + nc * (i64.bool (bin > 2))
    + d * (i64.bool (bin == 3))
  let is = map2 f bins offsets
  in scatter (#[scratch] copy xs) is xs

-- | Old radix sort with reduce.
def old_with_fuse_radix_sort [n] 't
                             (num_bits: i32)
                             (get_bit: i32 -> t -> i32)
                             (xs: [n]t) : [n]t =
  let iters = if n == 0 then 0 else (num_bits + 2 - 1) / 2
  in loop xs for i < iters do old_with_fuse_radix_sort_step xs get_bit (i * 2)

-- ==
-- entry: expected
-- input @ ../data/randomints_moderate_500MiB.in
entry expected = old_radix_sort i32.num_bits get_bit_i32

-- ==
-- input @ ../data/randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
entry main = old_with_fuse_radix_sort i32.num_bits get_bit_i32
