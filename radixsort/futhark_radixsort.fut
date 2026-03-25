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

-- entry: expected
-- input @ ../data/randomints_full_500MiB.in
-- input @ ../data/randomints_dense_500MiB.in
-- input @ ../data/randomints_moderate_500MiB.in
-- input @ ../data/randomints_sparse_500MiB.in
-- input @ ../data/randomints_empty_500MiB.in
-- entry expected = old_radix_sort i32.num_bits get_bit_i32

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
-- entry main = old_with_fuse_radix_sort i32.num_bits get_bit_i32

local
def exscan_i64 xs =
  map2 (-) (scan (+) 0i64 xs) xs

local
def exscan_i16 xs =
  map2 (-) (scan (+) 0i16 xs) xs

local
def get_bin 't
            (k: i32)
            (get_bit: i32 -> t -> i32)
            (digit_n: i32)
            (x: t) : i64 =
  i64.i32
  <| loop acc = 0
     for i < k do
       acc + (get_bit (digit_n + i) x << i)

local
def radix_sort_step_i16 [m] 't
                        (size: i64)
                        (get_bit: i32 -> t -> i32)
                        (digit_n: i32)
                        (block_offset: i64)
                        (xs: [m]t) : ([m]t, [16]i64, [16]i16) =
  let offset = block_offset * m
  let is =
    map2 (\j x ->
            if offset + j < size
            then get_bin 4 get_bit digit_n x
            else (1 << 4) - 1)
         (iota m)
         xs
  let bins = hist (+) 0 16 is (replicate m 1)
  let sorted =
    loop ys = copy xs
    for i in 0i32..<4i32 do
      let flags =
        map2 (\j x ->
                let b =
                  if offset + j < size
                  then get_bit (i + digit_n) x
                  else 1
                in (i16.i32 (1 ^ b), i16.i32 b))
             (iota m)
             ys
      let flags_scan =
        scan (\(a0, a1) (b0, b1) -> (a0 + b0, a1 + b1)) (0, 0) flags
      let os = hist (+) 0 1 (map (i64.i32 <-< (+ (-1)) <-< get_bit (i + digit_n)) xs) (rep 1)
      let o = os[0]
      let is =
        map2 (\f (o0, o1) ->
                if f.0 == 1 then o0 - 1 else o + o1 - 1)
             flags
             flags_scan
      in scatter ys (map i64.i16 is) (copy ys)
  let offsets = exscan_i16 bins
  in ( sorted
     , map i64.i16 bins
     , offsets
     )

local
def blocked_radix_sort_step [n] [m] 't
                            (size: i64)
                            (get_bit: i32 -> t -> i32)
                            (digit_n: i32)
                            (blocks: [n * m]t) =
  let (sorted_blocks, hist_blocks, offsets_blocks) =
    #[incremental_flattening(only_intra)]
    unflatten blocks
    |> map2 (radix_sort_step_i16 size get_bit digit_n) (iota n)
    |> unzip3
  let sorted = sized (n * m) (flatten sorted_blocks)
  let old_offsets = offsets_blocks
  let new_offsets =
    hist_blocks
    |> transpose
    |> flatten
    |> exscan_i64
    |> unflatten
    |> transpose
  let is =
    map2 (\i elem ->
            let bin =
              if i < size
              then get_bin 4 get_bit digit_n elem
              else (1 << 4) - 1
            let block_idx = i / m
            let new_offset = new_offsets[block_idx][bin]
            let old_block_offset = i64.i16 old_offsets[block_idx][bin]
            let old_offset = m * block_idx + old_block_offset
            let idx = (i - old_offset) + new_offset
            in idx)
         (iota (n * m))
         sorted
  in scatter (#[scratch] copy blocks) is sorted

def blocked_radix_sort [n] 't
                       (block: i16)
                       (num_bits: i32)
                       (get_bit: i32 -> t -> i32)
                       (xs: [n]t) : [n]t =
  let iters = if n == 0 then 0 else (num_bits + 4 - 1) / 4
  let block = i64.i16 block
  let n_blocks = if n == 0 then 0 else 1 + (n - 1) / block
  let empty = replicate (n_blocks * block) xs[0]
  in take n (loop ys = scatter empty (iota n) xs
             for i < iters do
               blocked_radix_sort_step n get_bit (i * 4) ys)

-- ==
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
entry main = blocked_radix_sort 1024 i32.num_bits get_bit_i32
